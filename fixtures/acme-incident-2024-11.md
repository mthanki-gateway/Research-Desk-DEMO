# Incident Post-Mortem: INC-2024-1183 — Payments Cutover Outage

## Summary

On 14 November 2024, the Acme payments service was unavailable or degraded for
2 hours and 47 minutes during a planned migration cutover. Approximately 18,400
customer transactions failed and required manual replay. No data was lost. No
customer funds were affected.

Severity was classified SEV-1. The incident commander was the payments platform
lead. This document is the blameless post-mortem, published 21 November 2024.

## Timeline

All times UTC on 14 November 2024.

02:00 — Planned cutover window opens. Traffic is shifted from the legacy
monolith to the new Kafka-backed payments service at 10% weight.

02:14 — Error rate on the new path rises to 4.2%. This is within the pre-agreed
1–5% tolerance for the first weight step, so the cutover proceeds.

02:31 — Weight increased to 50%. Error rate climbs to 22%. The on-call engineer
begins investigating but does not yet roll back, because the runbook's rollback
trigger was written as "sustained errors above 25%".

02:58 — Error rate reaches 61%. Rollback is initiated.

03:06 — Rollback completes at the traffic layer, but the legacy monolith cannot
absorb the returning load because its connection pool had been scaled down to 4
connections in preparation for decommissioning.

03:19 — Legacy monolith connection pool is scaled back to 40. Error rate begins
to fall.

04:47 — Error rate returns to baseline. Incident declared resolved.

## Root Cause

The immediate cause was connection pool exhaustion in the new payments service.
The service opened two independent connection pools against the same Postgres
instance — one from its ORM and one from its event-sourcing checkpoint writer —
and the configured pool size was interpreted as a per-pool limit rather than a
per-service limit.

Under 50% traffic weight, the two pools together requested 64 connections
against a database configured for a maximum of 40. Connection acquisition began
timing out, and because the retry policy used no jitter, the retries
synchronised into a thundering herd that made recovery slower.

The contributing cause was that the rollback trigger in the runbook (25%
sustained error rate) was set higher than the error rate at which customer
impact became material (approximately 8%).

## What Went Well

Detection was fast. The error-rate alert fired within 90 seconds of the 02:31
weight increase.

No data was lost. The event-sourced design meant failed transactions were
replayable from the log, and all 18,400 were successfully replayed by 09:00 the
same day.

The dual-write reconciliation job, which had been running for six weeks prior to
cutover, made it possible to verify that the two systems agreed before, during
and after the incident.

## What Went Poorly

The rollback was too slow. Eighteen minutes elapsed between the error rate
becoming materially harmful and the rollback being initiated, because the
runbook's threshold was wrong.

The legacy system had already been partially decommissioned. Scaling its
connection pool down before the new system had fully proven itself removed the
safety net that rollback depended on.

Nobody had load-tested the new service at 50% weight. Testing had been performed
at 10% and at 100% in a staging environment with a smaller database, and the
connection-pool interaction was invisible in both.

## Action Items

AI-1: Change the rollback trigger to 8% sustained error rate over 60 seconds.
Owner: payments platform lead. Completed 22 November 2024.

AI-2: Add a total-connections check at service startup that fails fast if the
sum of configured pool sizes exceeds the database maximum. Owner: platform
engineering. Completed 5 December 2024.

AI-3: Add jitter to all connection retry policies. Owner: platform engineering.
Completed 5 December 2024.

AI-4: Prohibit decommissioning of a legacy path until the replacement has run
at 100% weight for 14 consecutive days. Owner: engineering director. Policy
updated 1 December 2024.

AI-5: Extend load testing to include intermediate traffic weights. Owner: QA
lead. Target 31 January 2025. Status at time of writing: in progress.

## Lessons Recorded

A configured limit that is silently per-instance rather than per-service is a
recurring class of failure. The team has since audited three other services for
the same pattern and found one further instance, in the notifications service,
which was corrected without an incident.

Rollback capability is not a property of the new system. It is a property of the
old system still being able to take the load, and that property decays the
moment decommissioning begins.
