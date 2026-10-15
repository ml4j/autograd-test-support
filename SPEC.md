# Specification

Publish a standalone Java 25 JAR, org.ml4j:autograd-test-support:3.0.0-SNAPSHOT.
Compile dependencies are autograd and JUnit Jupiter; no tensor/backend dependency.
Consumers use test scope. autograd must never depend on this project.

AutogradContractTestBase<V extends AutogradValue<V, ?, ?>> is published from
src/main/java in org.ml4j.autograd.test. Protected hooks createValue(float, boolean),
multiply(V, V), and assertValue(float, V) adapt numerical fixtures without adding
production interfaces. Leaves must be fresh, mutually compatible, and represent
scalar numbers or uniform tensors. Multiplication must support second derivatives.
All backward tests use explicit compatible seeds; no implicit-seed, caching,
activation, broadcasting or resource policy is imposed.

Inherited tests exercise tracking, constants, shared/repeated graph inputs,
leaf accumulation, intermediate replacement, reset, stable gradient handles,
invalid roots/seeds, higher derivatives and parameter replacement isolation.
Two concrete suites in src/test/java run all contracts against the scalar factories.
Existing autograd unit tests and production code remain unchanged.

Tensor inheritance and duplicate-test consolidation are subsequent increments.
