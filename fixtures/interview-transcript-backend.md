# Interview Transcript — Senior Backend Engineer

Candidate: J. Okonkwo
Interviewer: R. Vance
Date: 14 March 2025
Duration: 52 minutes

This transcript is a fixture. It deliberately contains short back-referencing
turns ("elaborate on that", "why?") so that anaphora handling in chunking and
retrieval can be measured, not assumed.

## Introductions

**Interviewer:** Hi, thanks for making the time. I'm Rebecca, I lead the
platform team here.

**Candidate:** Good to meet you. I'm Jide, I've been doing backend work for
about nine years, mostly Python and Go, mostly payments and infrastructure.

**Interviewer:** Can you hear me okay? Great. Let's get into it.

## Payments Migration

**Interviewer:** Tell me about the most difficult migration you've run.

**Candidate:** The one that comes to mind is moving our payments service off a
Rails monolith onto an event-sourced service backed by Kafka. It took about
seven months end to end, and I led it.

**Interviewer:** Can you elaborate on that?

**Candidate:** Sure. The hard part wasn't the new service, it was cutting over
without losing transactions. We ran both systems in parallel for six weeks and
dual-wrote to each, with a reconciliation job comparing them nightly. Once the
reconciliation was clean for fourteen consecutive days we started shifting
traffic by weight.

**Interviewer:** And how did that go?

**Candidate:** Badly, the first time. We had an outage during the cutover. About
two hours and forty-five minutes of degraded service.

**Interviewer:** What caused it?

**Candidate:** Connection pool exhaustion. The new service opened two separate
pools against the same database — one from the ORM, one from the event
checkpoint writer — and we'd configured the pool size thinking it was a limit
per service. It wasn't, it was per pool. At fifty percent traffic weight the two
together asked for sixty-four connections against a database capped at forty.

**Interviewer:** Why didn't you catch that in testing?

**Candidate:** We tested at ten percent and at a hundred percent in staging. The
staging database was smaller but the pool config was the same, so at a hundred
percent it failed differently and we misdiagnosed it as a staging artefact. We
never tested the intermediate weights, which is exactly where it broke.

**Interviewer:** Interesting. And what did you change afterwards?

**Candidate:** Three things. We added a startup check that fails fast if the sum
of configured pool sizes exceeds the database maximum. We added jitter to the
retry policy, because the retries had synchronised into a thundering herd and
made recovery slower. And we changed the rollback threshold — it had been set at
twenty-five percent sustained errors, which was well above the point where
customers actually felt it.

**Interviewer:** What should it have been?

**Candidate:** Eight percent, over sixty seconds. That's what we set it to.

## On Rollback

**Interviewer:** You mentioned rollback. Say more about that.

**Candidate:** The thing I took away is that rollback isn't a property of the
new system. It's a property of the old system still being able to take the load.
We'd already scaled the monolith's connection pool down to four, because we were
getting ready to decommission it. So when we rolled back, the old path couldn't
absorb the traffic either, and that turned a fifteen-minute problem into an
hour.

**Interviewer:** That's a good insight. Did you write that down anywhere?

**Candidate:** It's in the post-mortem, and it became a policy — we don't
decommission a legacy path until the replacement has run at full weight for
fourteen days.

## Testing Practice

**Interviewer:** How do you think about testing on infrastructure work?

**Candidate:** I try to separate the things that are cheap and deterministic
from the things that aren't. Pure functions get exhaustive unit tests. Anything
touching a network or a clock gets a small number of integration tests and a lot
of care about what I'm actually asserting.

**Interviewer:** Can you give an example?

**Candidate:** Rate limiters are the classic one. The token bucket arithmetic is
pure — you can test refill and burst behaviour with a fake clock in about twenty
lines. But whether the limiter actually keeps you under a provider's quota is
not something a unit test can tell you, so that becomes a metric you watch in
production rather than a test you write.

**Interviewer:** And what about the things you can't test?

**Candidate:** You instrument them. If I can't test it I want to be able to see
it, and I'd rather ship a dashboard than a mock.

## Team and Process

**Interviewer:** How large was the team on the migration?

**Candidate:** Six engineers at peak, four for most of it. Two of us were on it
full time for the whole seven months.

**Interviewer:** How did you handle disagreement on approach?

**Candidate:** We wrote things down. If two people disagreed we'd write a short
document with the options and the trade-offs, and usually writing it resolved
it, because one option turned out to have a consequence nobody had said out
loud. When it didn't resolve it, I decided, and said why.

**Interviewer:** Any disagreements you got wrong?

**Candidate:** Yes. I argued against the dual-write reconciliation job for about
two weeks, because I thought it was over-engineering. It's the only reason we
could prove no data was lost during the outage. I was wrong and it mattered.

## Closing

**Interviewer:** What are you looking for in your next role?

**Candidate:** Work where correctness matters and where I can see the
consequences of what I build. And a team that writes things down.

**Interviewer:** That's all my questions. Anything you want to ask me?

**Candidate:** How does your team decide what to build next?
