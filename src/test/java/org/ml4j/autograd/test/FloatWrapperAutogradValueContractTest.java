package org.ml4j.autograd.test;

import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;
import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Runs the shared autograd contract against the factory-created FloatWrapperAutogradValue implementation. */
class FloatWrapperAutogradValueContractTest extends AutogradContractTestBase<FloatWrapperAutogradValue> {
    private final FloatWrapperAutogradValueFactory factory = new FloatWrapperAutogradValueFactoryImpl();

    @Override
    protected FloatWrapperAutogradValue createValue(float number, boolean tracked) {
        return factory.create(number, tracked);
    }

    @Override
    protected FloatWrapperAutogradValue multiply(FloatWrapperAutogradValue left, FloatWrapperAutogradValue right) {
        return left.mul(right);
    }

    @Override
    protected void assertValue(float expected, FloatWrapperAutogradValue actual) {
        assertEquals(expected, actual.data().get().floatValue(), 1e-5f);
    }
}
