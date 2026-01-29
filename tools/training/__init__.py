"""Training Integration for BlueprintPipeline.

Real-time feedback loop for streaming generated data to training systems.
"""

from .realtime_feedback import (
    RealtimeFeedbackLoop,
    TrainingSystemClient,
    DataStreamConfig,
    DataStreamProtocol,
    FeedbackMetrics,
)

__all__ = [
    "RealtimeFeedbackLoop",
    "TrainingSystemClient",
    "DataStreamConfig",
    "DataStreamProtocol",
    "FeedbackMetrics",
]
