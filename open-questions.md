# Onsite Open Questions

## Start here
1. **Walk me through how a researcher debugged the last reward hack.** Which steps map to our capabilities, which don't?
2. What does success look like for Dan at week 5?

## Labels and data
3. How were labels produced in prior experiments (programmatic, judge, hand)? Rollout- or span-level? How was quality measured?
4. Is there an existing labeled golden set to start from?
5. Is there a hack taxonomy or environment library that FDE findings feed into?

## Infrastructure and integration
6. Which RL framework do Baseten and its customers run (verl, TRL, OpenRLHF, in-house)?
7. Can we hook the trainer directly, or do we need a plugin the customer installs?
8. How is training sharded (FSDP, tensor parallel, pipeline parallel)?
9. Has Goodfire already hooked a trainer forward pass? Which layers, what overhead?
10. What latency overhead per RL step is acceptable?
11. What's the checkpoint retention policy today?
12. Which data stack does Baseten already run (Spark, Ray, Trino)? Reuse theirs.
13. Which experiment tracker do customers use? Is a Goodfire-hosted view allowed?

## Isolation, contracts, retention
14. What tenant isolation does Baseten provide, and where do training runs execute?
15. Does the pipeline run in Baseten's infra, the customer's VPC, or Goodfire's?
16. What can Goodfire aggregate or reuse across customers (metrics, behavior patterns, probes)?
17. What retention do customers expect? Does the DPA require a deletion SLA from day one?

## Later scope
18. Is serving-time monitoring on Baseten's roadmap, and on which engine?
