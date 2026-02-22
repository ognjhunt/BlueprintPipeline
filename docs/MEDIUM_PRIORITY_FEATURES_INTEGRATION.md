# Medium & Low Priority Features Integration Guide

This guide shows how to integrate the newly implemented features into the Genie Sim 3.0 pipeline.

## Overview of New Features

### Medium Priority (Section 3)
1. **Observability & Monitoring** - Real-time metrics collection
2. **Cost Tracking** - Per-scene cost analysis
3. **Parallel Scene Processing** - Batch processing for improved throughput
4. **Episode Diversity Metrics** - Quality analysis for training data

### Low Priority (Section 4)
1. **Soft Body Physics** - Deformable objects (cloth, rope, soft containers)
2. **Multi-Agent Scenes** - Collaborative manipulation with multiple robots
3. **Dynamic Scene Changes** - Moving obstacles, humans, dynamic lighting
4. **Real-Time Feedback Loop** - Stream data to training systems

---

## 1. Observability & Monitoring

### Usage in Pipeline Jobs

```python
# In any pipeline job (e.g., text-scene-adapter-job, simready-job, episode-generation-job)

from tools.metrics.pipeline_metrics import get_metrics

# Get metrics instance
metrics = get_metrics()

# Track job execution
with metrics.track_job("text-scene-adapter-job", scene_id):
    # Your job logic here
    process_scene(scene_id)

    # Track objects processed
    metrics.objects_processed.inc(num_objects, labels={"scene_id": scene_id})

# Track Gemini API calls
metrics.track_gemini_call(
    scene_id=scene_id,
    tokens_input=1000,
    tokens_output=500,
    operation="physics_estimation"
)

# Track Genie Sim jobs
metrics.track_geniesim_job(
    scene_id=scene_id,
    job_id=geniesim_job_id,
    duration_seconds=120.5,
    episode_count=500
)
```

### Integration Example: simready-job

```python
# simready-job/prepare_simready_assets.py

from tools.metrics.pipeline_metrics import get_metrics

def main():
    scene_id = os.getenv("SCENE_ID")
    metrics = get_metrics()

    with metrics.track_job("simready-job", scene_id):
        # Load manifest
        manifest = load_manifest(scene_id)

        # Process objects
        for obj in manifest["objects"]:
            try:
                # Estimate physics with Gemini
                with metrics.track_api_call("gemini", "estimate_physics", scene_id):
                    physics = estimate_physics_with_gemini(obj)

                # Track tokens
                metrics.track_gemini_call(
                    scene_id=scene_id,
                    tokens_input=obj_tokens_in,
                    tokens_output=obj_tokens_out,
                    operation="physics_estimation"
                )

                metrics.objects_processed.inc(labels={"scene_id": scene_id})

            except Exception as e:
                metrics.errors_total.inc(labels={
                    "job": "simready-job",
                    "scene_id": scene_id,
                    "error_type": type(e).__name__
                })
                raise
```

### Viewing Metrics

```bash
# For in-memory backend (development)
python -c "from tools.metrics.pipeline_metrics import get_metrics; \
           print(get_metrics().get_stats())"

# For Cloud Monitoring (production)
# View in Google Cloud Console:
# https://console.cloud.google.com/monitoring/metrics-explorer

# For Prometheus (production)
# Metrics exposed at http://localhost:8000/metrics
```

---

## 2. Cost Tracking

### Usage in Pipeline Jobs

```python
from tools.cost_tracking import get_cost_tracker

# Get tracker instance
tracker = get_cost_tracker()

# Track Gemini costs
tracker.track_gemini_call(
    scene_id=scene_id,
    tokens_in=1000,
    tokens_out=500,
    operation="physics_estimation"
)

# Track compute costs
tracker.track_compute(
    scene_id=scene_id,
    job_name="text-scene-adapter-job",
    duration_seconds=120.0,
    vcpu_count=2,
    memory_gb=4.0
)

# Track Genie Sim costs
tracker.track_geniesim_job(
    scene_id=scene_id,
    job_id=geniesim_job_id,
    episode_count=500,
    duration_seconds=300.0
)

# Get cost breakdown
breakdown = tracker.get_scene_cost(scene_id)
print(f"Total cost: ${breakdown.total:.4f}")
print(f"  Gemini: ${breakdown.gemini:.4f}")
print(f"  Genie Sim: ${breakdown.geniesim:.4f}")
print(f"  Cloud Run: ${breakdown.cloud_run:.4f}")

# Get optimization insights
insights = tracker.get_optimization_insights(scene_id)
for rec in insights["recommendations"]:
    print(f"💡 {rec['message']}")
```

### Integration Example: Full Pipeline

```python
# tools/run_local_pipeline.py (add cost tracking)

from tools.cost_tracking import get_cost_tracker

def run_pipeline(scene_id: str):
    tracker = get_cost_tracker()

    # Track each job's compute cost
    job_start = time.time()
    run_stage1_job(scene_id)
    job_duration = time.time() - job_start

    tracker.track_compute(
        scene_id=scene_id,
        job_name="text-scene-adapter-job",
        duration_seconds=job_duration,
        vcpu_count=2,
        memory_gb=4.0
    )

    # ... run other jobs ...

    # Report costs at the end
    breakdown = tracker.get_scene_cost(scene_id)
    print(f"\n💰 Cost Breakdown for {scene_id}:")
    print(f"  Total: ${breakdown.total:.4f}")
    print(f"  By component:")
    print(f"    Gemini API: ${breakdown.gemini:.4f}")
    print(f"    Genie Sim: ${breakdown.geniesim:.4f}")
    print(f"    Cloud Run: ${breakdown.cloud_run:.4f}")
    print(f"  By job:")
    for job_name, cost in breakdown.by_job.items():
        print(f"    {job_name}: ${cost:.4f}")
```

---

## 3. Parallel Scene Processing

### Usage for Batch Processing

```python
from tools.batch_processing import ParallelPipelineRunner
import asyncio

async def main():
    # Create runner
    runner = ParallelPipelineRunner(
        max_concurrent=5,  # Process 5 scenes at once
        retry_attempts=3,
        rate_limit_per_second=10.0
    )

    # Define processing function
    async def process_scene(scene_id: str):
        # Run pipeline for scene
        run_full_pipeline(scene_id)
        return {"scene_id": scene_id, "status": "success"}

    # Process batch
    scene_ids = ["scene_001", "scene_002", "scene_003", "scene_004", "scene_005"]

    results = await runner.process_batch(
        scene_ids=scene_ids,
        process_fn=process_scene,
        progress_callback=lambda completed, total: print(f"Progress: {completed}/{total}")
    )

    # Print results
    print(f"\n✅ Batch Results:")
    print(f"  Success: {results.success}/{results.total}")
    print(f"  Failed: {results.failed}/{results.total}")
    print(f"  Duration: {results.total_duration_seconds:.2f}s")
    print(f"  Success rate: {results.success/results.total:.1%}")

# Run
asyncio.run(main())
```

### Integration with Cloud Workflows

```python
# Create a new workflow: workflows/batch-genie-sim-pipeline.yaml

main:
  params: [event]
  steps:
    - extract_scenes:
        assign:
          - scene_ids: ${event.scene_ids}

    - run_parallel:
        call: http.post
        args:
          url: ${cloudRunJobUrl}/run-parallel
          auth:
            type: OIDC
          body:
            scene_ids: ${scene_ids}
            max_concurrent: 5
        result: batchResult

    - return_result:
        return: ${batchResult}
```

---

## 4. Episode Diversity Metrics

### Usage in Episode Generation

```python
from tools.quality.diversity_metrics import DiversityAnalyzer

# Create analyzer
analyzer = DiversityAnalyzer(
    workspace_bounds=(2.0, 2.0, 2.0),
    grid_resolution=0.1
)

# Analyze batch of episodes
episodes = [
    {
        "trajectory": [(0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.5)],
        "camera_poses": [{"position": [1.0, 1.0, 1.0], "orientation": [0, 0, 0, 1]}],
        "task_type": "pick_and_place",
        "object_interactions": ["mug", "table"],
        "success": True,
    },
    # ... more episodes
]

report = analyzer.analyze_batch(episodes)

# Print diversity scores
print(f"\n📊 Episode Diversity Report:")
print(f"  Overall Score: {report.overall_diversity_score:.2f}")
print(f"  Trajectory:")
print(f"    Variance: {report.trajectory.variance:.2f}")
print(f"    Workspace Coverage: {report.trajectory.workspace_coverage_pct:.1f}%")
print(f"  Visual:")
print(f"    Viewpoint Coverage: {report.visual.viewpoint_coverage:.2f}")
print(f"  Task:")
print(f"    Task Types: {len(report.task.task_type_distribution)}")

# Print recommendations
if report.recommendations:
    print(f"\n💡 Recommendations:")
    for rec in report.recommendations:
        print(f"  [{rec['severity'].upper()}] {rec['message']}")
```

### Integration Example: genie-sim-import-job

```python
# genie-sim-import-job/import_from_geniesim.py

from tools.quality.diversity_metrics import DiversityAnalyzer

def import_episodes(job_id: str):
    # Download episodes from Genie Sim
    episodes = download_episodes(job_id)

    # Analyze diversity
    analyzer = DiversityAnalyzer()
    report = analyzer.analyze_batch(episodes)

    # Add diversity report to quality certificate
    quality_cert = {
        "job_id": job_id,
        "episode_count": len(episodes),
        "diversity_score": report.overall_diversity_score,
        "diversity_report": report.to_dict(),
        "recommendations": report.recommendations,
    }

    # Save quality certificate
    save_quality_certificate(scene_id, quality_cert)

    # Log warnings if diversity is low
    if report.overall_diversity_score < 0.4:
        logger.warning(
            f"Low diversity score ({report.overall_diversity_score:.2f}) for {job_id}. "
            "Consider regenerating with more variation."
        )
```

---

## 5. Soft Body Physics Support

### Usage in Simready Job

```python
from tools.physics.soft_body import SoftBodyPhysics

# Create soft body detector
soft_body_physics = SoftBodyPhysics()

# Check if object should use soft body
obj_data = {
    "category": "towel",
    "material_name": "cotton",
}

if soft_body_physics.is_soft_body(obj_data):
    # Generate soft body properties
    props = soft_body_physics.generate_soft_body_properties(
        obj_data=obj_data,
        bounds={"size_m": [0.6, 0.4, 0.01], "volume_m3": 0.0024}
    )

    print(f"✨ Soft Body Detected:")
    print(f"  Type: {props.soft_body_type.value}")
    print(f"  Material: {props.material.value}")
    print(f"  Stiffness: {props.stiffness:.2f}")
    print(f"  Mass: {props.total_mass:.3f} kg")

    # Export to USD schema
    usd_schema = soft_body_physics.export_to_usd_schema(
        props=props,
        mesh_path="/path/to/mesh.usd"
    )

    # Add to object metadata
    obj_data["soft_body_properties"] = props.to_dict()
    obj_data["usd_soft_body_schema"] = usd_schema
```

### Integration Example: simready-job

```python
# simready-job/prepare_simready_assets.py

from tools.physics.soft_body import SoftBodyPhysics

def prepare_object_physics(obj: Dict[str, Any], bounds: Dict[str, Any]):
    # Try soft body first
    soft_body = SoftBodyPhysics()

    if soft_body.is_soft_body(obj):
        # Use soft body physics
        props = soft_body.generate_soft_body_properties(obj, bounds)

        return {
            "physics_type": "soft_body",
            "soft_body_properties": props.to_dict(),
            "mass_kg": props.total_mass,
        }
    else:
        # Use rigid body physics (existing code)
        return estimate_rigid_body_physics(obj, bounds)
```

---

## 6. Multi-Agent Scenes Support

### Usage in Episode Generation

```python
from tools.physics.multi_agent import (
    MultiAgentCoordinator,
    AgentConfiguration,
    CollisionAvoidanceStrategy,
    TaskAllocationStrategy
)

# Create coordinator
coordinator = MultiAgentCoordinator(
    collision_avoidance=CollisionAvoidanceStrategy.SIMPLE_SPHERE,
    min_separation_distance=0.3
)

# Add agents
agent1 = AgentConfiguration(
    agent_id="robot_1",
    robot_type="franka_panda",
    spawn_position=(-0.5, 0.0, 0.0),
    workspace_bounds=(1.0, 1.0, 1.0),
    priority=1
)

agent2 = AgentConfiguration(
    agent_id="robot_2",
    robot_type="ur5e",
    spawn_position=(0.5, 0.0, 0.0),
    workspace_bounds=(1.0, 1.0, 1.0),
    priority=2
)

coordinator.add_agent(agent1)
coordinator.add_agent(agent2)

# Set object positions
coordinator.set_object_positions({
    "obj_001": (-0.3, 0.5, 0.5),
    "obj_002": (0.3, 0.5, 0.5),
    "obj_003": (0.0, 0.6, 0.5),
})

# Allocate tasks
allocation = coordinator.allocate_tasks(
    objects=["obj_001", "obj_002", "obj_003"],
    strategy=TaskAllocationStrategy.NEAREST_NEIGHBOR
)

print(f"\n🤖 Multi-Agent Task Allocation:")
for agent_id, objects in allocation.items():
    print(f"  {agent_id}: {objects}")

# Check for conflicts
conflicts = coordinator.detect_workspace_conflicts()
if conflicts:
    print(f"\n⚠️  Workspace Conflicts Detected:")
    for conflict in conflicts:
        print(f"  {conflict['agent1']} <-> {conflict['agent2']}: {conflict['severity']}")

# Export configuration
config = coordinator.export_multi_agent_config()
```

### Integration Example: episode-generation-job

```python
# episode-generation-job/generate_episodes.py

from tools.physics.multi_agent import MultiAgentCoordinator

def generate_multi_agent_episode(scene_id: str, num_agents: int = 2):
    # Create coordinator
    coordinator = MultiAgentCoordinator()

    # Get suggested spawn positions
    positions = coordinator.suggest_spawn_positions(
        n_agents=num_agents,
        workspace_center=(0.0, 0.0, 0.5),
        workspace_radius=1.5
    )

    # Add agents
    for i, pos in enumerate(positions):
        agent = AgentConfiguration(
            agent_id=f"robot_{i}",
            robot_type="franka_panda",
            spawn_position=pos
        )
        coordinator.add_agent(agent)

    # Export to Isaac Sim config
    config = coordinator.export_multi_agent_config()

    # Use config in episode generation
    # ...
```

---

## 7. Dynamic Scene Changes Support

### Usage in Episode Generation

```python
from tools.physics.dynamic_scene import (
    DynamicSceneManager,
    DynamicObstacleType,
    HumanMotionPattern
)

# Create manager
manager = DynamicSceneManager()

# Add moving platform
platform = manager.create_conveyor_belt(
    obstacle_id="conveyor_1",
    position=(0.0, 0.5, 0.3),
    length=1.0,
    speed=0.1,
    direction=(1.0, 0.0, 0.0)
)

# Add human walking through scene
human = manager.create_walking_human(
    human_id="person_1",
    start_position=(-2.0, 0.0, 0.0),
    end_position=(2.0, 0.0, 0.0),
    walking_speed=1.2
)

# Enable dynamic lighting
manager.enable_dynamic_lighting(
    change_interval_seconds=10.0,
    ambient_range=(0.3, 1.0)
)

# Check for collisions with robot workspace
robot_workspace = ((-1.0, -1.0, 0.0), (1.0, 1.0, 1.5))
collisions = manager.detect_potential_collisions(
    robot_workspace=robot_workspace,
    time_horizon=10.0
)

if collisions:
    print(f"\n⚠️  Potential Collisions:")
    for collision in collisions:
        print(f"  {collision['type']} at t={collision['time']:.2f}s: {collision['position']}")

# Export configuration
config = manager.export_isaac_sim_config()
```

### Integration Example: episode-generation-job

```python
# episode-generation-job/generate_episodes.py

from tools.physics.dynamic_scene import DynamicSceneManager

def generate_dynamic_episode(scene_id: str):
    manager = DynamicSceneManager()

    # Add random human paths
    for i in range(2):
        start = np.random.uniform(-2, 2, 3)
        end = np.random.uniform(-2, 2, 3)

        manager.create_walking_human(
            human_id=f"person_{i}",
            start_position=tuple(start),
            end_position=tuple(end)
        )

    # Enable lighting variation
    manager.enable_dynamic_lighting()

    # Export for Isaac Sim
    config = manager.export_isaac_sim_config()

    # Add to episode metadata
    episode_metadata["dynamic_scene_config"] = config
```

---

## 8. Real-Time Feedback Loop

### Usage for Streaming to Training Systems

```python
from tools.training.realtime_feedback import (
    RealtimeFeedbackLoop,
    DataStreamConfig,
    DataStreamProtocol
)
import asyncio

async def main():
    # Configure streaming
    config = DataStreamConfig(
        protocol=DataStreamProtocol.HTTP_POST,
        endpoint_url="http://training-server:8000/episodes",
        api_key="your-api-key",
        batch_size=10,
        min_quality_score=0.7,
        max_wait_time_seconds=60.0
    )

    # Create feedback loop
    loop = RealtimeFeedbackLoop(config)

    # Start streaming
    await loop.start()

    # As episodes are generated, queue them
    for episode in generate_episodes():
        # Calculate quality score
        quality_score = calculate_quality(episode)

        # Queue episode (will be filtered if quality too low)
        queued = loop.queue_episode(episode, quality_score)

        if queued:
            print(f"✅ Episode queued (quality: {quality_score:.2f})")
        else:
            print(f"❌ Episode filtered (quality: {quality_score:.2f})")

    # Stop and flush remaining episodes
    await loop.stop()

    # Get statistics
    stats = loop.get_statistics()
    print(f"\n📊 Streaming Statistics:")
    print(f"  Episodes queued: {stats['episodes_queued']}")
    print(f"  Episodes sent: {stats['episodes_sent']}")
    print(f"  Episodes filtered: {stats['episodes_filtered']}")
    print(f"  Bytes sent: {stats['bytes_sent']:,}")

asyncio.run(main())
```

### Integration Example: genie-sim-import-job

```python
# genie-sim-import-job/import_from_geniesim.py

from tools.training.realtime_feedback import RealtimeFeedbackLoop, DataStreamConfig

async def import_and_stream(job_id: str):
    # Setup streaming
    config = DataStreamConfig(
        protocol=DataStreamProtocol.MESSAGE_QUEUE,
        endpoint_url=f"projects/{project_id}/topics/training-episodes",
        batch_size=50,
        min_quality_score=0.7
    )

    loop = RealtimeFeedbackLoop(config)
    await loop.start()

    try:
        # Download episodes from Genie Sim
        episodes = download_episodes(job_id)

        # Analyze quality and stream
        for episode in episodes:
            quality_score = calculate_quality_score(episode)
            loop.queue_episode(episode, quality_score)

        # Wait for all to be sent
        await asyncio.sleep(5)

    finally:
        await loop.stop()
```

---

## Complete Integration Example

Here's a complete example showing all features working together:

```python
# tools/run_enhanced_pipeline.py

import asyncio
from tools.metrics.pipeline_metrics import get_metrics
from tools.cost_tracking import get_cost_tracker
from tools.batch_processing import ParallelPipelineRunner
from tools.quality.diversity_metrics import DiversityAnalyzer
from tools.physics.soft_body import SoftBodyPhysics
from tools.physics.multi_agent import MultiAgentCoordinator
from tools.physics.dynamic_scene import DynamicSceneManager
from tools.training.realtime_feedback import RealtimeFeedbackLoop, DataStreamConfig

async def run_enhanced_pipeline(scene_ids: List[str]):
    """Run pipeline with all new features enabled."""

    # Initialize observability
    metrics = get_metrics()
    tracker = get_cost_tracker()

    # Initialize diversity analyzer
    diversity_analyzer = DiversityAnalyzer()

    # Initialize physics extensions
    soft_body = SoftBodyPhysics()

    # Setup streaming
    stream_config = DataStreamConfig(
        endpoint_url="http://training-server:8000/episodes",
        min_quality_score=0.7
    )
    feedback_loop = RealtimeFeedbackLoop(stream_config)
    await feedback_loop.start()

    try:
        # Process scenes in parallel
        runner = ParallelPipelineRunner(max_concurrent=5)

        async def process_scene(scene_id: str):
            with metrics.track_job("enhanced-pipeline", scene_id):
                # 1. Generate scene with soft bodies
                scene = prepare_scene(scene_id, soft_body)

                # 2. Multi-agent setup if needed
                if requires_multi_agent(scene):
                    coordinator = MultiAgentCoordinator()
                    setup_multi_agent(scene, coordinator)

                # 3. Dynamic scene setup
                dynamic_manager = DynamicSceneManager()
                add_dynamic_elements(scene, dynamic_manager)

                # 4. Generate episodes
                episodes = generate_episodes(scene)

                # 5. Analyze diversity
                diversity_report = diversity_analyzer.analyze_batch(episodes)

                # 6. Stream high-quality episodes
                for episode in episodes:
                    quality = calculate_quality(episode)
                    feedback_loop.queue_episode(episode, quality)

                # 7. Track costs
                track_scene_costs(scene_id, tracker, metrics)

                return {
                    "scene_id": scene_id,
                    "episodes": len(episodes),
                    "diversity_score": diversity_report.overall_diversity_score,
                }

        # Run batch
        results = await runner.process_batch(scene_ids, process_scene)

        # Print summary
        print(f"\n✅ Pipeline Complete:")
        print(f"  Scenes processed: {results.success}/{results.total}")
        print(f"  Duration: {results.total_duration_seconds:.2f}s")

        # Cost summary
        period_costs = tracker.get_period_cost(days=1)
        print(f"\n💰 Cost Summary:")
        print(f"  Total: ${period_costs['total']:.2f}")
        print(f"  Avg per scene: ${period_costs['avg_per_scene']:.2f}")

    finally:
        await feedback_loop.stop()

if __name__ == "__main__":
    scene_ids = ["scene_001", "scene_002", "scene_003"]
    asyncio.run(run_enhanced_pipeline(scene_ids))
```

---

## Next Steps

1. **Enable features in Genie Sim 3.0 workflows**: Update `workflows/genie-sim-export-pipeline.yaml` to use new features
2. **Configure monitoring**: Set up Cloud Monitoring dashboards (see `MONITORING_DASHBOARD.md`)
3. **Test with pilot scenes**: Run enhanced pipeline on 5-10 scenes and validate improvements
4. **Iterate based on metrics**: Use cost tracking and diversity metrics to optimize parameters

For more details, see individual module documentation in `tools/*/` directories.
