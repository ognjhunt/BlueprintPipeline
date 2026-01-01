"""Customer Success Metrics Tracking.

Tracks:
1. Scene delivery metrics (success rate, time to delivery)
2. Customer outcomes (did they train a policy? did it work?)
3. Pipeline performance (job success rates, bottlenecks)
4. Business metrics (revenue, retention)
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DeliveryStatus(str, Enum):
    """Scene delivery status."""
    PENDING = "pending"
    PROCESSING = "processing"
    QA_REVIEW = "qa_review"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CustomerFeedback(str, Enum):
    """Customer feedback categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class SceneDelivery:
    """Record of a scene delivery."""
    delivery_id: str
    scene_id: str
    customer_id: str

    # Status tracking
    status: DeliveryStatus = DeliveryStatus.PENDING
    created_at: str = ""
    delivered_at: Optional[str] = None

    # Timing
    processing_time_hours: float = 0.0
    time_in_qa_hours: float = 0.0
    total_time_hours: float = 0.0

    # Quality metrics
    qa_passed_first_try: bool = True
    qa_iterations: int = 1
    quality_score: float = 0.0

    # Scene details
    environment_type: str = "generic"
    object_count: int = 0
    episode_count: int = 0

    # Customer feedback
    customer_feedback: Optional[CustomerFeedback] = None
    feedback_notes: str = ""

    # Downstream usage
    policy_trained: bool = False
    policy_success_rate: Optional[float] = None
    real_world_tested: bool = False
    real_world_success_rate: Optional[float] = None

    def __post_init__(self):
        if not self.delivery_id:
            self.delivery_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

    def mark_delivered(self) -> None:
        """Mark as delivered."""
        self.status = DeliveryStatus.DELIVERED
        self.delivered_at = datetime.utcnow().isoformat() + "Z"

        # Calculate total time
        created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        delivered = datetime.fromisoformat(self.delivered_at.replace("Z", "+00:00"))
        self.total_time_hours = (delivered - created).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "scene_id": self.scene_id,
            "customer_id": self.customer_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "processing_time_hours": self.processing_time_hours,
            "time_in_qa_hours": self.time_in_qa_hours,
            "total_time_hours": self.total_time_hours,
            "qa_passed_first_try": self.qa_passed_first_try,
            "qa_iterations": self.qa_iterations,
            "quality_score": self.quality_score,
            "environment_type": self.environment_type,
            "object_count": self.object_count,
            "episode_count": self.episode_count,
            "customer_feedback": self.customer_feedback.value if self.customer_feedback else None,
            "feedback_notes": self.feedback_notes,
            "policy_trained": self.policy_trained,
            "policy_success_rate": self.policy_success_rate,
            "real_world_tested": self.real_world_tested,
            "real_world_success_rate": self.real_world_success_rate,
        }


@dataclass
class CustomerOutcome:
    """Track customer-level outcomes."""
    customer_id: str
    name: str
    email: str

    # Deliveries
    scenes_ordered: int = 0
    scenes_delivered: int = 0
    scenes_in_progress: int = 0

    # Success metrics
    policies_trained: int = 0
    policies_deployed: int = 0
    avg_sim_success_rate: float = 0.0
    avg_real_success_rate: float = 0.0
    avg_transfer_gap: float = 0.0

    # Satisfaction
    avg_feedback_score: float = 0.0  # 1-5 scale
    nps_score: Optional[int] = None  # -100 to 100

    # Revenue
    total_spent: float = 0.0
    lifetime_value: float = 0.0

    # Retention
    first_order_date: Optional[str] = None
    last_order_date: Optional[str] = None
    is_active: bool = True
    churned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "scenes_ordered": self.scenes_ordered,
            "scenes_delivered": self.scenes_delivered,
            "scenes_in_progress": self.scenes_in_progress,
            "policies_trained": self.policies_trained,
            "policies_deployed": self.policies_deployed,
            "avg_sim_success_rate": self.avg_sim_success_rate,
            "avg_real_success_rate": self.avg_real_success_rate,
            "avg_transfer_gap": self.avg_transfer_gap,
            "avg_feedback_score": self.avg_feedback_score,
            "nps_score": self.nps_score,
            "total_spent": self.total_spent,
            "lifetime_value": self.lifetime_value,
            "first_order_date": self.first_order_date,
            "last_order_date": self.last_order_date,
            "is_active": self.is_active,
            "churned": self.churned,
        }


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    period_start: str
    period_end: str

    # Volume
    scenes_processed: int = 0
    episodes_generated: int = 0

    # Success rates by job
    job_success_rates: Dict[str, float] = field(default_factory=dict)

    # Timing
    avg_processing_time_hours: float = 0.0
    p95_processing_time_hours: float = 0.0

    # Quality
    qa_first_pass_rate: float = 0.0
    avg_quality_score: float = 0.0

    # Failures
    total_failures: int = 0
    failure_by_stage: Dict[str, int] = field(default_factory=dict)
    top_failure_reasons: List[str] = field(default_factory=list)

    # Resource usage
    gpu_hours: float = 0.0
    compute_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "scenes_processed": self.scenes_processed,
            "episodes_generated": self.episodes_generated,
            "job_success_rates": self.job_success_rates,
            "avg_processing_time_hours": self.avg_processing_time_hours,
            "p95_processing_time_hours": self.p95_processing_time_hours,
            "qa_first_pass_rate": self.qa_first_pass_rate,
            "avg_quality_score": self.avg_quality_score,
            "total_failures": self.total_failures,
            "failure_by_stage": self.failure_by_stage,
            "top_failure_reasons": self.top_failure_reasons,
            "gpu_hours": self.gpu_hours,
            "compute_cost": self.compute_cost,
        }


class SuccessMetricsTracker:
    """Track and analyze success metrics."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir or "./metrics_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # In-memory caches
        self.deliveries: Dict[str, SceneDelivery] = {}
        self.customers: Dict[str, CustomerOutcome] = {}

        # Load existing data
        self._load_data()

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[METRICS] {msg}")

    def track_delivery(
        self,
        scene_id: str,
        customer_id: str,
        **kwargs
    ) -> SceneDelivery:
        """Start tracking a scene delivery."""
        delivery = SceneDelivery(
            delivery_id="",
            scene_id=scene_id,
            customer_id=customer_id,
            **kwargs
        )

        self.deliveries[delivery.delivery_id] = delivery
        self._save_delivery(delivery)

        # Update customer
        if customer_id not in self.customers:
            self.customers[customer_id] = CustomerOutcome(
                customer_id=customer_id,
                name=kwargs.get("customer_name", ""),
                email=kwargs.get("customer_email", ""),
            )

        self.customers[customer_id].scenes_ordered += 1
        self.customers[customer_id].scenes_in_progress += 1

        if not self.customers[customer_id].first_order_date:
            self.customers[customer_id].first_order_date = delivery.created_at

        self.customers[customer_id].last_order_date = delivery.created_at
        self._save_customer(self.customers[customer_id])

        self.log(f"Tracking delivery: {delivery.delivery_id} for scene {scene_id}")

        return delivery

    def update_delivery_status(
        self,
        delivery_id: str,
        status: DeliveryStatus,
        **kwargs
    ) -> SceneDelivery:
        """Update delivery status."""
        if delivery_id not in self.deliveries:
            self.deliveries[delivery_id] = self._load_delivery(delivery_id)

        delivery = self.deliveries[delivery_id]
        old_status = delivery.status
        delivery.status = status

        # Update fields
        for key, value in kwargs.items():
            if hasattr(delivery, key):
                setattr(delivery, key, value)

        if status == DeliveryStatus.DELIVERED:
            delivery.mark_delivered()

            # Update customer
            customer = self.customers.get(delivery.customer_id)
            if customer:
                customer.scenes_delivered += 1
                customer.scenes_in_progress -= 1
                self._save_customer(customer)

        self._save_delivery(delivery)
        self.log(f"Delivery {delivery_id}: {old_status.value} -> {status.value}")

        return delivery

    def record_customer_feedback(
        self,
        delivery_id: str,
        feedback: CustomerFeedback,
        notes: str = "",
    ) -> None:
        """Record customer feedback for a delivery."""
        if delivery_id not in self.deliveries:
            self.deliveries[delivery_id] = self._load_delivery(delivery_id)

        delivery = self.deliveries[delivery_id]
        delivery.customer_feedback = feedback
        delivery.feedback_notes = notes
        self._save_delivery(delivery)

        # Update customer average
        self._update_customer_metrics(delivery.customer_id)

        self.log(f"Recorded feedback for {delivery_id}: {feedback.value}")

    def record_training_outcome(
        self,
        delivery_id: str,
        policy_trained: bool = True,
        policy_success_rate: Optional[float] = None,
        real_world_tested: bool = False,
        real_world_success_rate: Optional[float] = None,
    ) -> None:
        """Record policy training and deployment outcomes."""
        if delivery_id not in self.deliveries:
            self.deliveries[delivery_id] = self._load_delivery(delivery_id)

        delivery = self.deliveries[delivery_id]
        delivery.policy_trained = policy_trained
        delivery.policy_success_rate = policy_success_rate
        delivery.real_world_tested = real_world_tested
        delivery.real_world_success_rate = real_world_success_rate
        self._save_delivery(delivery)

        # Update customer
        self._update_customer_metrics(delivery.customer_id)

        self.log(f"Recorded training outcome for {delivery_id}")

    def get_pipeline_metrics(
        self,
        days: int = 30,
    ) -> PipelineMetrics:
        """Get pipeline metrics for the last N days."""
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        # Filter deliveries in period
        period_deliveries = [
            d for d in self.deliveries.values()
            if d.created_at >= start.isoformat()
        ]

        metrics = PipelineMetrics(
            period_start=start.isoformat() + "Z",
            period_end=end.isoformat() + "Z",
            scenes_processed=len(period_deliveries),
        )

        if not period_deliveries:
            return metrics

        # Success rates
        delivered = [d for d in period_deliveries if d.status == DeliveryStatus.DELIVERED]
        failed = [d for d in period_deliveries if d.status == DeliveryStatus.FAILED]

        metrics.total_failures = len(failed)

        # Processing times
        times = [d.total_time_hours for d in delivered if d.total_time_hours > 0]
        if times:
            metrics.avg_processing_time_hours = statistics.mean(times)
            metrics.p95_processing_time_hours = (
                sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
            )

        # QA metrics
        qa_first = [d for d in delivered if d.qa_passed_first_try]
        metrics.qa_first_pass_rate = len(qa_first) / len(delivered) if delivered else 0

        # Quality scores
        scores = [d.quality_score for d in delivered if d.quality_score > 0]
        if scores:
            metrics.avg_quality_score = statistics.mean(scores)

        # Episode counts
        metrics.episodes_generated = sum(d.episode_count for d in delivered)

        return metrics

    def get_customer_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """Get dashboard data for a customer."""
        if customer_id not in self.customers:
            return {"error": f"Customer not found: {customer_id}"}

        customer = self.customers[customer_id]
        deliveries = [
            d for d in self.deliveries.values()
            if d.customer_id == customer_id
        ]

        return {
            "customer": customer.to_dict(),
            "recent_deliveries": [
                d.to_dict() for d in sorted(
                    deliveries,
                    key=lambda x: x.created_at,
                    reverse=True
                )[:10]
            ],
            "summary": {
                "total_scenes": len(deliveries),
                "delivered": sum(1 for d in deliveries if d.status == DeliveryStatus.DELIVERED),
                "in_progress": sum(1 for d in deliveries if d.status in [DeliveryStatus.PROCESSING, DeliveryStatus.QA_REVIEW]),
                "policies_trained": sum(1 for d in deliveries if d.policy_trained),
                "real_world_validated": sum(1 for d in deliveries if d.real_world_tested),
            }
        }

    def get_business_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get high-level business metrics."""
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        period_deliveries = [
            d for d in self.deliveries.values()
            if d.created_at >= start.isoformat()
        ]

        delivered = [d for d in period_deliveries if d.status == DeliveryStatus.DELIVERED]

        # Customer metrics
        active_customers = set(d.customer_id for d in period_deliveries)
        new_customers = set(
            c.customer_id for c in self.customers.values()
            if c.first_order_date and c.first_order_date >= start.isoformat()
        )

        # Feedback distribution
        feedback_dist = {}
        for d in delivered:
            if d.customer_feedback:
                fb = d.customer_feedback.value
                feedback_dist[fb] = feedback_dist.get(fb, 0) + 1

        # Success rates
        with_policy = [d for d in delivered if d.policy_trained]
        with_real_world = [d for d in delivered if d.real_world_tested]

        avg_policy_success = (
            statistics.mean(d.policy_success_rate for d in with_policy if d.policy_success_rate)
            if with_policy else 0
        )
        avg_real_success = (
            statistics.mean(d.real_world_success_rate for d in with_real_world if d.real_world_success_rate)
            if with_real_world else 0
        )

        return {
            "period": {
                "start": start.isoformat() + "Z",
                "end": end.isoformat() + "Z",
                "days": days,
            },
            "volume": {
                "scenes_ordered": len(period_deliveries),
                "scenes_delivered": len(delivered),
                "delivery_rate": len(delivered) / len(period_deliveries) if period_deliveries else 0,
            },
            "customers": {
                "active_customers": len(active_customers),
                "new_customers": len(new_customers),
                "total_customers": len(self.customers),
            },
            "quality": {
                "feedback_distribution": feedback_dist,
                "qa_first_pass_rate": sum(1 for d in delivered if d.qa_passed_first_try) / len(delivered) if delivered else 0,
            },
            "outcomes": {
                "policies_trained": len(with_policy),
                "policies_trained_rate": len(with_policy) / len(delivered) if delivered else 0,
                "real_world_tested": len(with_real_world),
                "avg_policy_success_rate": avg_policy_success,
                "avg_real_world_success_rate": avg_real_success,
                "avg_transfer_gap": avg_policy_success - avg_real_success if avg_real_success > 0 else None,
            },
        }

    def _update_customer_metrics(self, customer_id: str) -> None:
        """Update aggregated customer metrics."""
        if customer_id not in self.customers:
            return

        customer = self.customers[customer_id]
        deliveries = [
            d for d in self.deliveries.values()
            if d.customer_id == customer_id
        ]

        # Feedback score (1-5 scale)
        feedback_scores = {
            CustomerFeedback.EXCELLENT: 5,
            CustomerFeedback.GOOD: 4,
            CustomerFeedback.ACCEPTABLE: 3,
            CustomerFeedback.POOR: 2,
            CustomerFeedback.UNUSABLE: 1,
        }
        scores = [
            feedback_scores[d.customer_feedback]
            for d in deliveries
            if d.customer_feedback
        ]
        if scores:
            customer.avg_feedback_score = statistics.mean(scores)

        # Training outcomes
        trained = [d for d in deliveries if d.policy_trained]
        customer.policies_trained = len(trained)

        if trained:
            sim_rates = [d.policy_success_rate for d in trained if d.policy_success_rate]
            if sim_rates:
                customer.avg_sim_success_rate = statistics.mean(sim_rates)

        tested = [d for d in deliveries if d.real_world_tested]
        customer.policies_deployed = len(tested)

        if tested:
            real_rates = [d.real_world_success_rate for d in tested if d.real_world_success_rate]
            if real_rates:
                customer.avg_real_success_rate = statistics.mean(real_rates)

        customer.avg_transfer_gap = customer.avg_sim_success_rate - customer.avg_real_success_rate

        self._save_customer(customer)

    def _load_data(self) -> None:
        """Load existing data from disk."""
        deliveries_dir = self.data_dir / "deliveries"
        if deliveries_dir.exists():
            for path in deliveries_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    delivery = self._delivery_from_dict(data)
                    self.deliveries[delivery.delivery_id] = delivery
                except Exception:
                    pass

        customers_dir = self.data_dir / "customers"
        if customers_dir.exists():
            for path in customers_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    customer = self._customer_from_dict(data)
                    self.customers[customer.customer_id] = customer
                except Exception:
                    pass

    def _save_delivery(self, delivery: SceneDelivery) -> None:
        """Save delivery to disk."""
        path = self.data_dir / "deliveries" / f"{delivery.delivery_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(delivery.to_dict(), indent=2))

    def _load_delivery(self, delivery_id: str) -> SceneDelivery:
        """Load delivery from disk."""
        path = self.data_dir / "deliveries" / f"{delivery_id}.json"
        data = json.loads(path.read_text())
        return self._delivery_from_dict(data)

    def _save_customer(self, customer: CustomerOutcome) -> None:
        """Save customer to disk."""
        path = self.data_dir / "customers" / f"{customer.customer_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(customer.to_dict(), indent=2))

    def _delivery_from_dict(self, data: Dict[str, Any]) -> SceneDelivery:
        """Create delivery from dict."""
        return SceneDelivery(
            delivery_id=data["delivery_id"],
            scene_id=data["scene_id"],
            customer_id=data["customer_id"],
            status=DeliveryStatus(data.get("status", "pending")),
            created_at=data.get("created_at", ""),
            delivered_at=data.get("delivered_at"),
            processing_time_hours=data.get("processing_time_hours", 0.0),
            time_in_qa_hours=data.get("time_in_qa_hours", 0.0),
            total_time_hours=data.get("total_time_hours", 0.0),
            qa_passed_first_try=data.get("qa_passed_first_try", True),
            qa_iterations=data.get("qa_iterations", 1),
            quality_score=data.get("quality_score", 0.0),
            environment_type=data.get("environment_type", "generic"),
            object_count=data.get("object_count", 0),
            episode_count=data.get("episode_count", 0),
            customer_feedback=CustomerFeedback(data["customer_feedback"]) if data.get("customer_feedback") else None,
            feedback_notes=data.get("feedback_notes", ""),
            policy_trained=data.get("policy_trained", False),
            policy_success_rate=data.get("policy_success_rate"),
            real_world_tested=data.get("real_world_tested", False),
            real_world_success_rate=data.get("real_world_success_rate"),
        )

    def _customer_from_dict(self, data: Dict[str, Any]) -> CustomerOutcome:
        """Create customer from dict."""
        return CustomerOutcome(
            customer_id=data["customer_id"],
            name=data.get("name", ""),
            email=data.get("email", ""),
            scenes_ordered=data.get("scenes_ordered", 0),
            scenes_delivered=data.get("scenes_delivered", 0),
            scenes_in_progress=data.get("scenes_in_progress", 0),
            policies_trained=data.get("policies_trained", 0),
            policies_deployed=data.get("policies_deployed", 0),
            avg_sim_success_rate=data.get("avg_sim_success_rate", 0.0),
            avg_real_success_rate=data.get("avg_real_success_rate", 0.0),
            avg_transfer_gap=data.get("avg_transfer_gap", 0.0),
            avg_feedback_score=data.get("avg_feedback_score", 0.0),
            nps_score=data.get("nps_score"),
            total_spent=data.get("total_spent", 0.0),
            lifetime_value=data.get("lifetime_value", 0.0),
            first_order_date=data.get("first_order_date"),
            last_order_date=data.get("last_order_date"),
            is_active=data.get("is_active", True),
            churned=data.get("churned", False),
        )
