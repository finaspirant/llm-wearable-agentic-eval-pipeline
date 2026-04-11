"""Shared tool definitions for all agent implementations.

Provides a common set of tools that all framework implementations
(LangGraph, CrewAI, AutoGen, OpenAI SDK) use — ensuring the
benchmark comparison measures framework differences, not tool
availability differences.
"""
import logging

logger = logging.getLogger(__name__)

# TODO(Day 19): Define shared tools
#   - check_heart_rate(user_id) → SensorReading
#   - check_location(user_id) → GPSCoordinate
#   - send_notification(user_id, message, urgency) → bool
#   - check_calendar(user_id, time_range) → list[Event]
#   - assess_privacy_level(context) → PrivacyLevel
#   - escalate_to_human(reason, context) → EscalationResult
