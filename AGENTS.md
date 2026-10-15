# Agent instructions

Read SPEC.md and PLAN.md before changing this standalone Java 25 Maven project.

- Keep reusable JUnit bases in src/main/java/org/ml4j/autograd/test.
- Keep scalar contract runners in src/test/java; production scalar classes stay in autograd.
- Keep dependencies flowing toward autograd; never add this project to autograd.
- JUnit is compile-scoped here; downstream consumers use test scope.
- Preserve established contracts and existing coverage. Agree contract changes first.
- Keep fixtures deterministic and fresh; avoid tensor/backend dependencies.
- Document public types and protected fixture hooks with concise Javadoc.
- Run ./mvnw clean verify; report inherited tests actually executed by concrete suites.
- Update SPEC.md and PLAN.md with implemented decisions and verification.
