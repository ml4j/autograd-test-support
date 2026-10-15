# autograd-test-support

Reusable JUnit 5 autograd contract tests, independent of numerical backends.
Requires Java 25; builds independently with the Maven 3.9.12 Wrapper.

```sh
./mvnw clean verify
```

Requires `org.ml4j:autograd:3.0.0-SNAPSHOT` in your Maven repository. For local
work, run `./mvnw install` in autograd first.

Consumers add `org.ml4j:autograd-test-support:3.0.0-SNAPSHOT` with test scope and
extend `org.ml4j.autograd.test.AutogradContractTestBase` in test sources.
Implement `createValue`, `multiply`, and `assertValue` for compatible scalar or
uniform tensor fixtures. JUnit is deliberately a compile dependency of this JAR.

This project also runs the contract against raw and wrapped scalar implementations.
Scalar production classes and existing unit tests remain in autograd, which does
not depend on this project. Tensor-specific tests belong in tensors-test-support.
