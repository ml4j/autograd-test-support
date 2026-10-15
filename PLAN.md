# Development plan

- [x] Establish standalone Maven JAR and Maven 3.9.12 Wrapper.
- [x] Publish generic AutogradContractTestBase with numerical fixture hooks.
- [x] Add raw and wrapped scalar contract runners using existing factories.
- [x] Preserve existing autograd production code and tests.
- [x] Verify both scalar runners and the existing autograd suite.
- [ ] Adapt tensors-test-support to inherit the generic base; validate both backends.
- [ ] Review equivalent tensor tests for consolidation without losing coverage.

Tensor suite migration is a subsequent increment; it is unchanged in this step.

Verification: `./mvnw -o -B clean verify` passed 16 inherited tests (eight
per scalar implementation). The existing autograd suite passed all 108 tests;
its current artifact was installed locally for the support build.
