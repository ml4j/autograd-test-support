# Development plan

- [x] Establish standalone Maven JAR and Maven 3.9.12 Wrapper.
- [x] Publish generic AutogradContractTestBase with numerical fixture hooks.
- [x] Add raw and wrapped scalar contract runners using existing factories.
- [x] Preserve existing autograd production code and tests.
- [x] Verify both scalar runners and the existing autograd suite.
- [x] Adapt tensors-test-support to inherit the generic base; validate both backends.
- [x] Review equivalent tensor tests for consolidation without losing coverage.

Tensor inheritance is implemented: the tensor base supplies uniform 2-by-3
fixtures. Existing overlapping tensor checks retain sum-reduction coverage.

Verification: `./mvnw -o -B clean verify` passed 16 inherited tests (eight
per scalar implementation). The existing autograd suite passed all 108 tests;
its current artifact was installed locally for the support build.

Tensor reactor verification: `./mvnw -o -B clean install` passed 56 DL4J and
43 DJL tests, including all eight generic contract checks per backend.
