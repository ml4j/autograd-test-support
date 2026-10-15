package org.ml4j.autograd.test;

import org.ml4j.autograd.scalar.FloatAutogradValue;
import org.ml4j.autograd.scalar.FloatAutogradValueFactory;
import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Runs the shared autograd contract against the factory-created FloatAutogradValue implementation. */
class FloatAutogradValueContractTest extends AutogradContractTestBase<FloatAutogradValue> {
    private final FloatAutogradValueFactory factory = new FloatAutogradValueFactoryImpl();

    @Override
    protected FloatAutogradValue createValue(float number, boolean tracked) {
        return factory.create(number, tracked);
    }

    @Override
    protected FloatAutogradValue multiply(FloatAutogradValue left, FloatAutogradValue right) {
        return left.mul(right);
    }

    @Override
    protected void assertValue(float expected, FloatAutogradValue actual) {
        assertEquals(expected, actual.data().get(), 1e-5f);
    }
}
