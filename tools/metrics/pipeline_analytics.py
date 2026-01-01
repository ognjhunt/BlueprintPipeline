"""Pipeline Analytics - Convenience Functions.

High-level functions for tracking pipeline runs and customer outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .success_metrics import (
    SuccessMetricsTracker,
    SceneDelivery,
    DeliveryStatus,
    CustomerFeedback,
)


# Global tracker instance
_tracker: Optional[SuccessMetricsTracker] = None


def get_tracker() -> SuccessMetricsTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        data_dir = Path(os.getenv("METRICS_DATA_DIR", "./metrics_data"))
        _tracker = SuccessMetricsTracker(data_dir=data_dir)
    return _tracker


def track_pipeline_run(
    scene_id: str,
    customer_id: str,
    environment_type: str = "generic",
    **kwargs
) -> str:
    """Track a new pipeline run.

    Call this when starting to process a scene.

    Args:
        scene_id: Scene identifier
        customer_id: Customer identifier
        environment_type: Type of environment
        **kwargs: Additional delivery properties

    Returns:
        delivery_id for tracking

    Example:
        delivery_id = track_pipeline_run(
            scene_id="kitchen_001",
            customer_id="acme_robotics",
            environment_type="kitchen",
        )
    """
    tracker = get_tracker()
    delivery = tracker.track_delivery(
        scene_id=scene_id,
        customer_id=customer_id,
        environment_type=environment_type,
        **kwargs
    )
    return delivery.delivery_id


def update_pipeline_status(
    delivery_id: str,
    status: str,
    **kwargs
) -> None:
    """Update pipeline run status.

    Args:
        delivery_id: Delivery identifier
        status: New status (pending, processing, qa_review, delivered, failed)
        **kwargs: Additional updates

    Example:
        update_pipeline_status(delivery_id, "processing")
        update_pipeline_status(delivery_id, "qa_review", quality_score=0.85)
        update_pipeline_status(delivery_id, "delivered", episode_count=500)
    """
    tracker = get_tracker()
    tracker.update_delivery_status(
        delivery_id=delivery_id,
        status=DeliveryStatus(status),
        **kwargs
    )


def track_scene_delivery(
    delivery_id: str,
    object_count: int = 0,
    episode_count: int = 0,
    quality_score: float = 0.0,
    qa_passed_first_try: bool = True,
    processing_time_hours: float = 0.0,
) -> None:
    """Mark a scene as delivered with metrics.

    Args:
        delivery_id: Delivery identifier
        object_count: Number of objects in scene
        episode_count: Number of episodes generated
        quality_score: Overall quality score (0-1)
        qa_passed_first_try: Whether QA passed on first attempt
        processing_time_hours: Total processing time

    Example:
        track_scene_delivery(
            delivery_id="abc123",
            object_count=15,
            episode_count=500,
            quality_score=0.87,
            processing_time_hours=2.5
        )
    """
    tracker = get_tracker()
    tracker.update_delivery_status(
        delivery_id=delivery_id,
        status=DeliveryStatus.DELIVERED,
        object_count=object_count,
        episode_count=episode_count,
        quality_score=quality_score,
        qa_passed_first_try=qa_passed_first_try,
        processing_time_hours=processing_time_hours,
    )


def track_customer_feedback(
    delivery_id: str,
    feedback: str,
    notes: str = "",
) -> None:
    """Record customer feedback for a delivery.

    Args:
        delivery_id: Delivery identifier
        feedback: Feedback rating (excellent, good, acceptable, poor, unusable)
        notes: Additional feedback notes

    Example:
        track_customer_feedback(
            delivery_id="abc123",
            feedback="excellent",
            notes="Perfect physics, trained policy worked first try"
        )
    """
    tracker = get_tracker()
    tracker.record_customer_feedback(
        delivery_id=delivery_id,
        feedback=CustomerFeedback(feedback),
        notes=notes,
    )


def track_training_outcome(
    delivery_id: str,
    policy_success_rate: float,
    real_world_tested: bool = False,
    real_world_success_rate: Optional[float] = None,
) -> None:
    """Record policy training and deployment outcomes.

    Args:
        delivery_id: Delivery identifier
        policy_success_rate: Success rate in simulation (0-1)
        real_world_tested: Whether policy was tested in real world
        real_world_success_rate: Real-world success rate if tested

    Example:
        # After simulation training
        track_training_outcome(
            delivery_id="abc123",
            policy_success_rate=0.92
        )

        # After real-world deployment
        track_training_outcome(
            delivery_id="abc123",
            policy_success_rate=0.92,
            real_world_tested=True,
            real_world_success_rate=0.78
        )
    """
    tracker = get_tracker()
    tracker.record_training_outcome(
        delivery_id=delivery_id,
        policy_trained=True,
        policy_success_rate=policy_success_rate,
        real_world_tested=real_world_tested,
        real_world_success_rate=real_world_success_rate,
    )


def get_dashboard_data(
    customer_id: Optional[str] = None,
    days: int = 30,
) -> Dict[str, Any]:
    """Get dashboard data.

    Args:
        customer_id: Optional customer ID for customer-specific dashboard
        days: Number of days for metrics period

    Returns:
        Dashboard data dict

    Example:
        # Business metrics dashboard
        data = get_dashboard_data(days=30)

        # Customer-specific dashboard
        data = get_dashboard_data(customer_id="acme_robotics")
    """
    tracker = get_tracker()

    if customer_id:
        return tracker.get_customer_dashboard(customer_id)
    else:
        return {
            "business_metrics": tracker.get_business_metrics(days=days),
            "pipeline_metrics": tracker.get_pipeline_metrics(days=days).to_dict(),
        }


def get_success_summary() -> Dict[str, Any]:
    """Get a quick summary of success metrics.

    Returns:
        Summary dict with key metrics

    Example:
        summary = get_success_summary()
        print(f"Delivery rate: {summary['delivery_rate']:.1%}")
        print(f"Avg transfer gap: {summary['avg_transfer_gap']:.1%}")
    """
    tracker = get_tracker()
    business = tracker.get_business_metrics(days=30)
    pipeline = tracker.get_pipeline_metrics(days=30)

    return {
        "scenes_delivered_30d": business["volume"]["scenes_delivered"],
        "delivery_rate": business["volume"]["delivery_rate"],
        "qa_first_pass_rate": pipeline.qa_first_pass_rate,
        "avg_quality_score": pipeline.avg_quality_score,
        "policies_trained_rate": business["outcomes"]["policies_trained_rate"],
        "avg_sim_success_rate": business["outcomes"]["avg_policy_success_rate"],
        "avg_real_success_rate": business["outcomes"]["avg_real_world_success_rate"],
        "avg_transfer_gap": business["outcomes"]["avg_transfer_gap"],
        "active_customers": business["customers"]["active_customers"],
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pipeline Analytics")
    subparsers = parser.add_subparsers(dest="command")

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Get dashboard data")
    dash_parser.add_argument("--customer-id", help="Customer ID for specific dashboard")
    dash_parser.add_argument("--days", type=int, default=30)

    # Summary
    subparsers.add_parser("summary", help="Get quick summary")

    # Track
    track_parser = subparsers.add_parser("track", help="Track a delivery")
    track_parser.add_argument("--scene-id", required=True)
    track_parser.add_argument("--customer-id", required=True)
    track_parser.add_argument("--environment-type", default="generic")

    args = parser.parse_args()

    if args.command == "dashboard":
        data = get_dashboard_data(
            customer_id=args.customer_id,
            days=args.days,
        )
        print(json.dumps(data, indent=2, default=str))

    elif args.command == "summary":
        summary = get_success_summary()
        print("\nBlueprintPipeline Success Metrics (Last 30 Days)")
        print("=" * 50)
        print(f"Scenes Delivered:     {summary['scenes_delivered_30d']}")
        print(f"Delivery Rate:        {summary['delivery_rate']:.1%}")
        print(f"QA First Pass Rate:   {summary['qa_first_pass_rate']:.1%}")
        print(f"Avg Quality Score:    {summary['avg_quality_score']:.2f}")
        print(f"Policies Trained:     {summary['policies_trained_rate']:.1%}")
        print(f"Avg Sim Success:      {summary['avg_sim_success_rate']:.1%}")
        print(f"Avg Real Success:     {summary['avg_real_success_rate']:.1%}")
        if summary['avg_transfer_gap']:
            print(f"Avg Transfer Gap:     {summary['avg_transfer_gap']:.1%}")
        print(f"Active Customers:     {summary['active_customers']}")

    elif args.command == "track":
        delivery_id = track_pipeline_run(
            scene_id=args.scene_id,
            customer_id=args.customer_id,
            environment_type=args.environment_type,
        )
        print(f"Tracking delivery: {delivery_id}")

    else:
        parser.print_help()
