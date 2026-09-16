# Acme Corporation — Engineering Handbook (Revision 11, June 2025)

## About This Handbook

This handbook is the single reference for how engineering works at Acme. It
supersedes the Platform Wiki, which was retired in March 2025 after an audit
found forty-one pages describing systems that no longer existed.

Revision 11 incorporates the remediation items from the November 2024 payments
cutover incident, the Q1 FY2025 reliability review, and the data platform
migration completed in May 2025.

Where this handbook disagrees with a team's own runbook, the runbook is
authoritative for the specific procedure and this handbook is authoritative for
policy. Disagreements should be reported to the Platform team rather than
resolved locally.

## Engineering Organisation

Engineering is 214 people across eleven teams as of 1 June 2025, up from 178 at
the end of FY2024. The org is grouped into four pillars: Product Engineering,
Platform, Data, and Security.

Product Engineering is the largest pillar at 96 people and owns everything a
customer touches directly. Platform is 54 people and owns compute, networking,
CI/CD and developer tooling. Data is 38 and owns the warehouse, streaming
infrastructure and the machine learning platform. Security is 26 and owns
identity, application security and compliance.

Every team has a designated Technical Lead and a Product Manager. Teams larger
than twelve people are split at the next planning cycle; this rule was
introduced after the Payments team reached nineteen and its incident response
became unmanageable.

## Service Ownership

Every production service has exactly one owning team, recorded in the service
catalogue. A service without an owner is deleted after ninety days of notice.
This is not a threat: four services were deleted under this policy in FY2024 and
none were missed.

Ownership means the team is paged for the service, approves changes to it, and
is accountable for its error budget. Ownership does not mean the team is the
only group permitted to change it — Acme practises open contribution, and any
engineer may raise a pull request against any repository.

The owning team must review within two working days. Reviews that stall beyond
that are escalated to the pillar lead, not to the individual reviewer.

## The Service Catalogue

The catalogue lives at catalogue.internal.acme.example and is generated from
`service.yaml` files committed alongside each service. A service that is not in
the catalogue cannot be deployed to production; the deploy pipeline checks for
the file and fails closed.

Each entry records the owning team, the on-call rotation, the tier, the runbook
URL, the data classification, and every downstream dependency. Dependencies are
declared manually. An attempt to infer them from network traffic in 2024 was
abandoned because it could not distinguish a health check from a real call.

## Service Tiers

Services are classified Tier 1 through Tier 4. The tier determines the
availability target, the review requirements and the on-call expectations.

Tier 1 services are customer-facing and revenue-critical: the API gateway, the
authentication service, the payments service and the core data plane. Their
target is 99.95% monthly availability, which allows roughly 22 minutes of
downtime per month.

Tier 2 services are customer-facing but degrade gracefully: search, reporting,
the notifications service and the admin console. Their target is 99.9%, or about
43 minutes per month.

Tier 3 services are internal and business-hours critical: the deploy pipeline,
the metrics store and the feature flag service. Tier 4 covers everything else,
including batch jobs and development tooling, and carries no formal target.

## Error Budgets

A Tier 1 service with a 99.95% target has an error budget of 0.05% of the month,
or about 22 minutes. Consuming the budget is expected and is not a failure;
consuming it repeatedly is a signal that reliability work has been deprioritised
too far.

When a service exhausts its error budget in a calendar month, feature work on
that service stops until the budget recovers. Only reliability work, security
fixes and customer-blocking bugs ship. This is enforced by the owning team's
lead, not by tooling.

The payments service exhausted its budget in November 2024 for the first time
since the policy was introduced. Feature work was frozen for six weeks.

## On-Call

Every Tier 1 and Tier 2 service has a 24/7 on-call rotation staffed by its
owning team. Rotations are one week long and hand over at 10:00 on Wednesdays,
deliberately mid-week so a handover is never adjacent to a weekend.

A rotation must have at least six people. Below six, the rotation is merged with
an adjacent team's or the service is demoted to Tier 3. Five-person rotations
were tried in 2023 and produced a measurable spike in voluntary attrition.

On-call engineers are compensated at a flat rate per week plus time in lieu for
any night page. Being paged at night entitles the engineer to start the
following day at noon or take the day off, at their discretion, and this is not
negotiated with the manager.

## Paging Policy

An alert may page a human only if it is urgent, actionable and specific. An
alert that is merely interesting goes to a dashboard. An alert that is urgent but
not actionable is a bug in the alert, not a signal to be louder.

Every page must link to a runbook section that says what to do. Pages without a
runbook link are auto-closed by the alerting system after fifteen minutes and
filed as a defect against the owning team.

The 2024 alert audit removed 312 of 511 alert rules. Page volume fell 61% and
mean time to acknowledge improved from 11 minutes to 4. No incident was missed as
a result, which was the risk everyone worried about and nobody measured
beforehand.

## Incident Severity

Incidents are SEV1 through SEV4. SEV1 means customer-visible loss of a Tier 1
service or any confirmed data loss. SEV2 means significant degradation with a
workaround. SEV3 is limited impact, and SEV4 is a near miss with no customer
effect.

SEV1 and SEV2 incidents require a written post-mortem within five working days.
SEV3 requires a short note. SEV4 requires only a catalogue entry, but teams are
encouraged to write them up anyway — the November 2024 payments incident was
preceded by two SEV4s with the same root cause, neither of which was written up.

## Incident Command

Every SEV1 and SEV2 has three named roles: Incident Commander, Communications
Lead and Operations Lead. The Incident Commander does not debug. This is the
rule most often broken and the one that most reliably lengthens an incident.

The Commander's job is to maintain the timeline, decide between options and
protect the Operations Lead from interruption. If the Commander is typing
commands, nobody is doing the Commander's job.

Communications Lead posts an update every fifteen minutes to the status page,
even when the update is "no change". Silence is read by customers as a longer
outage than it is.

## Post-Mortems Are Blameless

A post-mortem describes what happened, what contributed, and what will change.
It does not name the individual who typed a command. The person who triggered an
incident is almost never its cause; the cause is a system that allowed a
reasonable action to have an unreasonable effect.

Every post-mortem must list contributing factors, not a single root cause.
Complex systems fail for several reasons at once, and the search for one root
cause consistently stops at the most visible contributor rather than the most
important one.

Action items must have an owner and a date. Post-mortems with unowned actions are
returned by the reliability review.

## The November 2024 Payments Incident

On 14 November 2024, a cutover from the legacy payments monolith to the new
payments service caused a 47-minute partial outage affecting approximately 12% of
transactions. Full detail is in the dedicated incident document; this section
records only the handbook changes that resulted.

Four policies changed. Connection pool sizing is now derived from a documented
formula rather than chosen per service. Rollback thresholds must be set to the
point of customer impact, not to an arbitrary error rate. Load testing must cover
intermediate traffic weights, not only 10% and 100%. And decommissioning a
legacy path may not begin until the replacement has served full traffic for two
weeks.

## Connection Pool Sizing

Pool size is calculated as (expected concurrent requests × average database calls
per request) divided by the number of service instances, plus a headroom factor
of 1.3. The result is capped at the database's connection limit divided by the
number of services sharing it.

The failure mode this prevents is arithmetic rather than subtle: two independent
services each sized for 64 connections against a database permitting 40 in total
will exhaust it at roughly half their design load. This is exactly what happened
in November 2024.

Every service must declare its pool size in `service.yaml`. The deploy pipeline
sums declared pools per database and fails if the total exceeds the limit. This
check was added in December 2024 and has failed six deploys since, every one of
them correctly.

## Deployment Pipeline

All production deploys go through the pipeline. There is no manual path, and the
production Kubernetes credentials required to bypass it are held only by the
break-glass rotation.

A deploy runs unit tests, integration tests, a container build, a security scan,
a staging deploy, smoke tests, and then a progressive production rollout. The
median pipeline duration is 14 minutes; the 95th percentile is 31 minutes,
dominated by integration test flakiness that remains unresolved.

Deploys are blocked on Fridays after 14:00 and during the two weeks around the
fiscal year end. Emergency fixes may override this with a pillar lead's approval,
which is recorded automatically.

## Progressive Rollout

Production rollouts move through 1%, 10%, 50% and 100% of traffic. Each stage
holds for a minimum bake time: five minutes at 1% and 10%, fifteen minutes at
50%. Automated rollback triggers on error rate, latency and a service-specific
custom metric.

The 50% stage exists because of the November 2024 incident, where a defect
appeared only at intermediate load. Before that, rollouts went 10% to 100%
directly, and a class of resource-contention failures was structurally invisible.

## Feature Flags

Feature flags are for decoupling deploy from release, not for permanent
configuration. Every flag has an owner and an expiry date recorded at creation.

Flags older than 90 days appear on a weekly report to the owning team. Flags
older than 180 days are removed by the Platform team with one week's notice. At
the time of writing there are 87 active flags, down from 341 in January 2025.

A flag that has been at 100% for a month is not a flag, it is a branch nobody
deleted. Cleaning these up removed 4,100 lines of dead code in Q1 FY2025.

## Testing Policy

Every service must have unit tests covering its business logic and integration
tests covering its external contracts. Coverage is measured but no threshold is
enforced, because a coverage target reliably produces tests written to satisfy
the target.

Integration tests run against real dependencies in staging, not mocks. Mocked
integration tests passed continuously through the November 2024 incident and
would not have caught it, because the mock did not have a connection limit.

Flaky tests are quarantined automatically after three non-deterministic failures
in a week and must be fixed or deleted within ten working days. A quarantined
test that is neither fixed nor deleted is deleted.

## Code Review

Every change requires one approving review. Changes to Tier 1 services, to
authentication, or to anything handling payment data require two, at least one
from outside the authoring team.

Reviews should focus on correctness, clarity and blast radius. Style is enforced
by the formatter and should never be the subject of a review comment; if a style
question is arguable, the formatter configuration is the place to argue it.

The target is a first response within four working hours. Measured median is 2.1
hours, and the 90th percentile is 9 hours, which is worse than it looks because
the tail is concentrated in two teams.

## Branching and Release

Acme uses trunk-based development. Branches are short-lived and merged within two
days; anything longer is a design problem being deferred.

Releases are cut from trunk automatically on every merge. There are no release
branches, and there is no code freeze other than the fiscal-year-end window.

Reverting is preferred to fixing forward during an incident. A revert is a known
state; a fix under time pressure is a new and untested one.

## Observability Standards

Every service emits structured logs, metrics and traces. Logs are JSON, one event
per line, and must include a request ID propagated from the gateway.

Metrics follow the RED method for request-driven services — Rate, Errors,
Duration — and the USE method for resources: Utilisation, Saturation, Errors.
Custom metrics are permitted but must be documented in the service catalogue.

Traces are sampled at 1% by default and at 100% for requests that error. The
error-path sampling was added in 2024 and immediately made a class of
intermittent failures diagnosable that had previously been invisible in
aggregate.

## Log Retention and Cost

Logs are retained hot for 14 days and cold for 400 days. The 400-day figure is
driven by the compliance requirement to answer questions about the prior fiscal
year, not by engineering need.

Logging cost reached $71,000 per month in Q3 FY2024, at which point a review
found that 38% of volume came from three debug statements left enabled in the
notifications service. Current spend is approximately $24,000 per month against a
budget of $30,000.

Teams see their own logging spend on a weekly report. Showing teams the number
reduced volume by more than any policy did.

## Data Platform Architecture

The warehouse is Snowflake, fed by Kafka for streaming and Airflow for batch. The
May 2025 migration consolidated three separate warehouses — one per pillar — into
a single account with per-team databases.

Raw data lands in a bronze layer exactly as received, with no transformation.
Silver applies schema, types and deduplication. Gold contains the modelled tables
that analytics and reporting read. No consumer may read bronze directly, and this
is enforced by grants rather than convention.

The migration moved 340 TB and ran for eleven days. Two tables were found to have
been silently empty since 2023; nobody had noticed because the dashboards reading
them defaulted to zero rather than erroring.

## Streaming Infrastructure

Kafka runs as a managed cluster with 24 brokers across three availability zones.
Topics are provisioned through code review, not through a console, and every
topic declares a retention period and an owning team.

Default retention is seven days. Topics requiring longer must justify it, because
retention is the single largest driver of cluster cost and the default is almost
always sufficient for replay.

Consumer lag is alerted at 5 minutes for Tier 1 consumers and 30 minutes
otherwise. Lag alerts page only when they are also increasing; a flat lag of six
minutes is a capacity conversation, not an incident.

## Data Quality Checks

Every gold table has freshness, volume and null-rate checks. A check failure
opens a ticket against the owning team and marks the table as stale in the
catalogue, which propagates a warning to every downstream dashboard.

Freshness checks compare the maximum timestamp to the expected schedule with a
tolerance of one interval. Volume checks compare row counts to a 28-day trailing
median, alerting outside a band of ±40%, which is wide enough to survive normal
weekly seasonality.

The ±40% band was ±20% until March 2025, when it was widened after month-end
processing tripped it every month for a year and everyone learned to ignore it.

## Personally Identifiable Information

PII is classified at three levels. Level 1 is directly identifying: name, email,
postal address, phone number. Level 2 is indirectly identifying, such as IP
address or device identifier. Level 3 is sensitive: payment details, government
identifiers and anything about health.

Level 3 data may not enter the warehouse at all. Payment details live only in the
payments service and its vault, and analytics receives a tokenised reference.
This restriction is why the November 2024 incident, despite its severity, had no
data exposure component.

Every column containing Level 1 or Level 2 data is tagged in the catalogue.
Untagged columns in a table known to contain PII fail the weekly compliance scan.

## Data Retention and Deletion

Customer data is retained for the life of the contract plus 90 days. Deletion
requests are honoured within 30 days across all systems including backups, which
is the requirement that makes them expensive.

Backups are the hard part. Acme uses a crypto-shredding approach: per-customer
encryption keys are destroyed on deletion, rendering backup copies unreadable
without restoring and rewriting 400 days of archives.

Deletion is verified by a quarterly audit that samples ten deleted customers and
searches for their identifiers across every system. The Q1 FY2025 audit found one
residual copy in a Tier 4 analytics sandbox, which was remediated and the sandbox
brought into the deletion pipeline.

## Identity and Access

All human access is through single sign-on with hardware-backed multi-factor
authentication. Password authentication to internal systems was removed entirely
in August 2024.

Service-to-service authentication uses short-lived mTLS certificates issued by
the internal certificate authority, with a 24-hour lifetime. Long-lived service
account keys are prohibited; the four that remained in 2024 were eliminated by
February 2025.

Production database access requires an approved just-in-time grant with a stated
reason and a maximum duration of four hours. Grants are logged and reviewed
weekly. Median grants per week fell from 34 to 9 after the reason field was made
mandatory, which suggests most of the original volume was habit.

## Secrets Management

Secrets live in the vault and are injected at runtime. Secrets in environment
variables baked into images, in configuration files, or in CI variables are
prohibited.

Every repository is scanned on push and on a nightly full-history sweep. A
detected secret triggers automatic revocation before it triggers a notification —
the notification is useless if the credential is still valid when it arrives.

Seventeen secrets were detected in 2024, of which fourteen were test credentials
and three were real. All three were revoked within four minutes of detection.

## Vulnerability Management

Container images are scanned at build and re-scanned daily, because a vulnerable
dependency is usually disclosed after the image is built rather than before.

Critical vulnerabilities in internet-facing services must be remediated within 7
days, high within 30, medium within 90. The clock starts at disclosure, not at
detection, which is deliberately unforgiving: a scanner that is a week behind
does not extend the deadline.

Exceptions require sign-off from the Security pillar lead and expire after 30
days. There are currently three active exceptions, all for a transitive
dependency with no available patched version.

## Machine Learning Platform

The ML platform supports feature engineering, training, model registry and
serving. It is deliberately not an experimentation platform; notebooks are a
local concern and are not productionised directly.

Every model in production has a registered version, a documented training
dataset, an owner, and an evaluation report. Models without an evaluation report
cannot be promoted, which has blocked promotion four times and been overridden
zero times.

Model serving is behind the same gateway as every other service and carries the
same tier, alerting and error budget rules. A model is a service that happens to
have weights.

## Model Evaluation

Every model is evaluated on a held-out set that the training pipeline is
structurally unable to see. The split is by customer and by time, not at random —
random splits leak information between training and evaluation whenever records
from one customer or one week appear in both.

Offline metrics are necessary and not sufficient. Every model promoted to
production runs first as a shadow, scoring live traffic without affecting it, for
a minimum of one week.

The recommendation model's offline AUC improved by 0.04 in February 2025 while
its shadow click-through rate fell. The model was not promoted. This is the
clearest example anyone has of why shadow deployment exists.

## Retrieval and Search Quality

The document search system uses hybrid retrieval: dense vector similarity fused
with BM25 lexical matching. The fusion is reciprocal rank fusion with a constant
of 60, chosen because results were insensitive to values between 20 and 100.

Retrieval quality is measured on a golden set of labelled questions, tracking
recall at 5 and 10, mean reciprocal rank and NDCG. Changes to chunking, embedding
model or fusion require a measured comparison against the previous configuration.

Chunk size is 500 characters with 80 characters of overlap, arrived at
empirically. Larger chunks retrieved more context and diluted the embedding;
smaller chunks split sentences and lost the antecedents of pronouns.

## Embedding Model Policy

The production embedding model is versioned in the catalogue and may not be
changed without re-embedding the entire corpus. Mixed-version embeddings in a
single index produce silently wrong similarity, which is the worst kind of wrong
because nothing errors.

Re-embedding 340 TB is a multi-day operation, so model changes are batched and
performed at most twice a year. The current model produces 768-dimensional
vectors.

Vectors are normalised at write time, which makes cosine similarity a dot product
and removes an entire class of bug in which magnitude leaks into a similarity
score.

## API Design Standards

APIs are REST over HTTPS with JSON. GraphQL was evaluated in 2023 and rejected —
not on technical merit, but because the caching and rate-limiting infrastructure
was built around URL-addressable resources and the migration cost exceeded the
benefit.

Every endpoint is versioned in the path. Breaking changes require a new version
and a twelve-month overlap during which both are served. Non-breaking changes are
additive only: a field may be added, never removed or retyped.

Pagination is cursor-based. Offset pagination is prohibited, because a page-3
request against a list that changed underneath returns duplicated or skipped
records and nobody notices until a customer reconciles totals.

## Rate Limiting

Rate limits are per API key and per endpoint class, enforced at the gateway. The
default is 1,000 requests per minute for read endpoints and 100 for write
endpoints.

Limits return HTTP 429 with a `Retry-After` header. Clients that ignore it are
progressively backed off, and clients that ignore that are suspended with an
email to the account owner.

The gateway uses a sliding window rather than a fixed one. Fixed windows permit
double the intended rate across a boundary, which a customer discovered in 2023
and reported, to their credit, rather than exploiting.

## Idempotency

Every write endpoint accepts an `Idempotency-Key` header. Keys are stored for 24
hours with the response they produced, and a repeated key returns the stored
response rather than performing the operation again.

This is mandatory for payment operations and strongly recommended everywhere
else. During the November 2024 incident, idempotency keys prevented duplicate
charges when clients retried against a degraded service, and it is the single
reason that incident was a reliability event rather than a financial one.

## Mobile Release Process

The mobile applications release on a fixed two-week train. Features that miss the
train wait for the next one; the train does not wait.

Every release spends at least three days in staged rollout, starting at 5% of
users. Crash-free session rate must stay above 99.5% to proceed, measured over a
minimum of 20,000 sessions.

Server-side changes must remain compatible with the two most recent mobile
releases, which is six weeks. Analysis of the install base shows 94% of active
users update within six weeks; the remaining tail is addressed by a forced-update
prompt after twelve.

## Accessibility

All customer-facing interfaces target WCAG 2.2 Level AA. This is a requirement,
not an aspiration, and it is checked in CI with automated auditing plus a manual
review each quarter.

Automated tools catch roughly 30% of real accessibility defects. The other 70%
require keyboard navigation testing and a screen reader, which is why the manual
review exists and why it cannot be replaced by more tooling.

Colour is never the sole carrier of meaning. Every status indicator pairs colour
with a shape, an icon or text, which also makes screenshots legible in the
monochrome printouts the finance team still uses.

## Internationalisation

All user-facing strings are externalised. String concatenation to build sentences
is prohibited, because word order differs between languages and the resulting
sentence is ungrammatical in most of them.

Dates, times, numbers and currencies are formatted by locale at render time and
stored in ISO 8601 and minor currency units respectively. Storing a formatted
date is a bug that surfaces months later in a different timezone.

Acme currently ships in English, German, French, Spanish, Japanese and Brazilian
Portuguese. Adding a language is roughly six weeks: two for translation, two for
layout adjustment, and two for the defects nobody predicted.

## Hiring Process

The engineering interview is four stages: a recruiter screen, a technical screen,
a panel, and a values conversation. Total candidate time is capped at six hours,
including preparation.

There is no whiteboard algorithm round. It was removed in 2022 after an analysis
of 140 hires found no correlation between performance in that round and
performance after twelve months, and a measurable correlation with candidate
withdrawal.

The panel is a code reading exercise, a system design discussion, and a
conversation about a real problem the candidate has solved. Interviewers write
their assessment before the debrief, not during it, to reduce anchoring.

## Levelling and Progression

Engineering has six levels from L1 to L6, with L7 and above reserved for a small
number of principal roles. The ladder is public, including the compensation bands
for every level.

Progression is evidence-based and reviewed twice yearly. The evidence is
collected continuously by the engineer and their manager, because a review that
depends on recalling six months of work over-weights the last six weeks.

There is an equivalent management ladder with equal compensation at equal level.
Moving between the two does not reset level and is explicitly expected to happen
more than once in a career.

## Onboarding

New engineers ship a change to production in their first week. The change is
small and real, and the point is to exercise the whole path — environment,
review, pipeline, deploy, observe — while someone is sitting with them.

The first month has a named buddy separate from the manager. The buddy answers
questions that feel too small to ask a manager, which is most of the questions
that actually block someone.

Onboarding is measured by time to first production change and by a survey at 30
and 90 days. Median time to first change is 3.5 days, down from 11 in 2023 after
the development environment was containerised.

## Development Environment

The development environment is a single `docker compose up`. Anything requiring
a manual setup step is a bug against the Platform team, and this rule is why
onboarding time fell.

Local development uses containerised dependencies, not shared staging ones.
Shared development databases were retired in 2024; they were a constant source of
interference and made every local failure ambiguous.

Compute-heavy work uses remote development machines, which are provisioned on
demand and destroyed after seven days of inactivity. Average utilisation is 31%,
which is understood and accepted as the cost of instant availability.

## Documentation Expectations

Every service has a README explaining what it does, who owns it, how to run it
locally and how to debug it in production. A README that only says how to build
is not a README.

Architecture decisions are recorded as ADRs committed alongside the code. An ADR
states the context, the decision, the alternatives considered and the
consequences. It is never edited after acceptance; a changed decision is a new
ADR that supersedes the old one, because the historical record is the value.

Comments explain why, not what. A comment restating the code is worse than no
comment because it will drift out of date and then actively mislead.

## Technical Debt

Technical debt is tracked in the same backlog as feature work, with the same
prioritisation. A separate debt backlog is a place debt goes to be forgotten.

Each team allocates at least 20% of capacity to debt and reliability. This is a
floor, not a target, and teams under error-budget freeze allocate considerably
more.

Debt items must state the cost of not fixing them in terms someone outside
engineering can evaluate: incident risk, delivery slowdown, or direct spend. An
item that cannot be stated that way is usually a preference rather than debt.

## Build and CI Infrastructure

CI runs on ephemeral runners provisioned per job. Persistent runners were retired
in 2024 after a build contaminated by a previous job's cache produced a passing
test suite against the wrong artefact.

Build caching is content-addressed and shared across the organisation. Cache hit
rate is 78% and each percentage point is worth approximately 40 engineer-hours
per month, which is the argument that funds the infrastructure.

Total CI spend is $46,000 per month. This is reported alongside the time saved,
because a cost figure without the corresponding benefit invites the wrong
decision.

## Dependency Management

Dependencies are pinned to exact versions and updated by automation that opens a
pull request per dependency. Automatic merging is enabled for patch versions
where the test suite passes.

New dependencies require justification in review. The bar is higher for anything
in a Tier 1 service and higher again for anything that executes at request time
rather than at build time.

Acme maintains an internal mirror of public registries. The mirror exists for
availability and for the ability to answer, after a supply chain incident
elsewhere, exactly which versions were installed on which day.

## Capacity Planning

Capacity is planned quarterly from the product roadmap and the trailing growth
rate, with a target of 40% headroom at peak on every Tier 1 service.

Forty percent sounds generous and is not: peak traffic is 3.2 times median, and
the largest single-day spike in FY2024 was 5.1 times median during a customer's
promotional event that Acme was not told about in advance.

Autoscaling handles variation within a day. It does not handle a step change in
baseline, which is what capacity planning is for; conflating the two is how teams
end up autoscaling into a database connection limit.

## Cost Management

Infrastructure spend was $2.1 million for FY2024 and is tracked per team and per
service. Every service in the catalogue has an attributed cost, updated monthly.

Untagged resources are charged to the pillar rather than the team, which creates
exactly enough discomfort to get them tagged. Untagged spend fell from 19% to 3%
within two quarters of this policy.

Cost review is quarterly and looks for the largest absolute savings first. Cost
work is otherwise prone to spending an engineer-week to save forty dollars a
month, which is a poor trade that feels productive.

## Disaster Recovery

The recovery time objective for Tier 1 services is four hours and the recovery
point objective is five minutes. These are tested, not asserted.

A full regional failover exercise runs twice yearly with production traffic. The
May 2025 exercise completed in 2 hours 51 minutes, within objective, and found
two services with hardcoded region endpoints that had passed every previous
review.

Backups are restored monthly to a scratch environment and verified by checksum
and by a query returning expected row counts. A backup that has never been
restored is a hypothesis, not a backup.

## Third-Party Vendors

Any vendor processing customer data requires a security review, a signed data
processing agreement, and an entry in the vendor register with a named internal
owner.

Vendor availability is assumed to be worse than advertised. Every integration
with an external service must degrade gracefully, and services whose failure
would take down a Tier 1 path require a documented fallback.

The payment processor integration has a documented fallback to a secondary
processor, tested quarterly. The email provider does not, which is an accepted
risk recorded in the register: delayed email is tolerable in a way that failed
payment is not.

## Meetings and Communication

Engineering runs on written communication. Any meeting with more than four
attendees requires an agenda circulated beforehand and notes published
afterwards.

There is one protected no-meeting day per week, Wednesday, which applies across
all four pillars. It was Thursday until 2024 and moved because it was adjacent to
Friday's deploy freeze and the two combined into a dead half-week.

Decisions made verbally are not decisions until they are written down. This
sounds bureaucratic and prevents the specific failure where four people leave a
room with three understandings.

## Remote and Distributed Work

Engineering is distributed across seven countries and eleven time zones.
Synchronous meetings are scheduled in a four-hour overlap window; anything
outside it is recorded.

Documents are written to be read by someone who was not there. This is the single
highest-leverage habit for distributed work and the hardest to maintain, because
writing for the absent reader is slower than writing for the present one.

Every team publishes a weekly written update. The update is for the organisation,
not for the manager, and reporting to the organisation produces a noticeably more
honest document.

## Open Source

Acme contributes upstream to the projects it depends on, and contributing a fix
upstream is preferred to maintaining a fork. Forks are permitted for urgency and
must have a plan to converge.

Releasing Acme code as open source requires legal review and a committed
maintainer. Projects without a maintainer are archived rather than left to accrue
unanswered issues, which is worse for users than never publishing.

The internal policy is that any code with no Acme-specific business logic should
be open source by default. Three libraries have been released under this policy
since 2023.

## Security Incident Response

A suspected security incident is reported to the Security pillar immediately,
before any investigation. Investigating first destroys evidence and delays
containment, and the reporter is never penalised for a false alarm.

Security incidents follow the same severity scale as reliability incidents but
have a separate escalation path that includes Legal and Communications from the
outset. Regulatory notification timelines start at discovery, not at
confirmation, which is why the first call is to Security and not to a colleague.

Tabletop exercises run quarterly. The Q2 FY2025 exercise simulated a compromised
CI credential and found that revocation was documented but had never been
practised, taking 40 minutes rather than the assumed 5.

## Change Management

All production changes are recorded automatically by the pipeline, including who
approved them and what tests ran. There is no separate change request process,
because a manual record duplicated from an automatic one is a record that
disagrees with itself.

High-risk changes — schema migrations, network configuration, identity changes —
require a written plan including a rollback procedure that has been tested. An
untested rollback is not a rollback.

Schema migrations are expand-contract in every case: add the new structure,
migrate, switch reads, then remove the old structure in a later release. A
single-step migration cannot be rolled back once traffic has written to the new
shape.

## Database Standards

The primary transactional database is PostgreSQL. Services own their schemas and
no service reads another's tables directly; cross-service reads go through APIs.

Migrations run automatically at deploy and must be backward compatible with the
previous release, because a progressive rollout has both versions serving traffic
simultaneously for up to twenty minutes.

Long-running migrations are prohibited in the deploy path. A migration expected to
exceed thirty seconds runs as a separate backfill job with progress reporting and
the ability to pause.

## Caching Policy

Caches are an optimisation, never a correctness mechanism. A system that is
incorrect with a cold cache is incorrect, and will eventually run with a cold
cache.

Every cache entry has an explicit TTL. Unbounded caches are prohibited, and a
cache without an eviction policy is an out-of-memory error with a delay.

Cache invalidation is by TTL expiry wherever tolerable and by explicit
invalidation only where staleness is genuinely unacceptable. Explicit
invalidation across service boundaries is a distributed-systems problem that
looks like a one-line change.

## Queue and Retry Semantics

All asynchronous work goes through durable queues with at-least-once delivery.
Consumers must be idempotent, because at-least-once means duplicates will happen
and "rare" is not "never" at Acme's volume.

Retries use exponential backoff with jitter. Backoff without jitter
synchronises every client into a thundering herd, which is a contributing factor
in the November 2024 incident and in two earlier incidents nobody connected at
the time.

Every queue has a dead letter queue, and every dead letter queue has an alert and
an owner. Unmonitored dead letter queues are where data goes to be lost quietly.

## Timeouts

Every network call has an explicit timeout. A call without one inherits a default
measured in minutes, which converts a slow dependency into an exhausted thread
pool and a cascading failure.

Timeouts are set from the caller's latency budget, not from the callee's typical
response time. A request with a two-second budget cannot afford a five-second
timeout on any single hop, regardless of how quickly that hop usually responds.

Timeout budgets are propagated in a header so downstream services know how long
the caller is still willing to wait, and can abandon work that is already
pointless.

## Performance Targets

Tier 1 API endpoints target a 95th percentile latency of 300 milliseconds and a
99th percentile of 800 milliseconds, measured at the gateway.

Measurement is at the gateway rather than inside the service deliberately.
Service-internal timing consistently reports better numbers than customers
experience, because it excludes queueing, TLS and the network.

Averages are not tracked. An average latency hides the tail entirely, and the
tail is what customers describe when they say the product feels slow.

## Load Testing

Every Tier 1 service is load tested before launch and after any change to its
resource profile. Tests run against staging with production-shaped data, because
synthetic uniform data does not produce realistic cache or index behaviour.

Tests must cover intermediate load levels, not only the target. The November 2024
failure occurred at approximately 50% of design load and would have been found by
any test that looked there.

Results are recorded in the catalogue alongside the configuration they were run
against. A load test result without its configuration is not reproducible and
therefore not evidence.

## Deprecation Policy

Deprecating an internal API requires notice proportional to the migration cost:
one quarter minimum, longer where consumers must change data models.

Deprecation notices name a date, an owner and a migration guide. A notice without
a date is ignored, and every deprecation without a hard date in Acme's history
has run past two years.

Usage is measured before removal. A deprecated endpoint with live traffic is not
removed regardless of the announced date; the date is a commitment to be ready,
not permission to break a caller.

## Glossary

**Error budget** — the amount of unreliability a service may accumulate in a
month while still meeting its target.

**Expand-contract** — a migration pattern that adds new structure before removing
old, so both versions can run simultaneously.

**Golden set** — a curated collection of labelled questions used to measure
retrieval quality over time.

**RED method** — Rate, Errors, Duration: the three metrics tracked for every
request-driven service.

**Shadow deployment** — running a new model or service against live traffic
without acting on its output, to compare behaviour before promotion.

**Tier** — a service's criticality classification, which determines its
availability target and on-call requirements.

**USE method** — Utilisation, Saturation, Errors: the three metrics tracked for
every resource.
