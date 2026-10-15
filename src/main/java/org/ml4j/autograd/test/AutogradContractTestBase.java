package org.ml4j.autograd.test;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.impl.ParameterImpl;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reusable checks for autograd tracking, traversal and differentiable gradients.
 *
 * <p>Subclasses supply fresh leaves representing a scalar number, multiplication,
 * and numerical assertions. A tensor fixture may use scalar-shaped tensors or
 * uniformly filled tensors with compatible seeds. No broadcasting, implicit seed,
 * caching, activation or native-resource policy is assumed.
 *
 * <p>Consumers extend this class in their test sources and depend on this JAR with
 * test scope. Each test creates its own graph; fixture methods must preserve the
 * production value contracts and must not share mutable gradient state.
 *
 * @param <V> the autograd value type under test
 */
public abstract class AutogradContractTestBase<V extends AutogradValue<V, ?, ?>> {
    /** Constructor for concrete contract suites. */
    protected AutogradContractTestBase() {}

    /**
     * Creates a fresh leaf representing the given number, with no stored gradient.
     * @param number the numerical contents, or uniform tensor element value
     * @param tracked whether gradients are required
     * @return a fresh compatible leaf
     */
    protected abstract V createValue(float number, boolean tracked);

    /**
     * Multiplies compatible values without changing their contents.
     * @param left the left operand
     * @param right the right operand
     * @return a differentiable product supporting second derivatives
     */
    protected abstract V multiply(V left, V right);

    /**
     * Checks numerical contents without changing the value or gradient state.
     * @param expected the expected scalar or uniform element value
     * @param actual the value to inspect
     */
    protected abstract void assertValue(float expected, V actual);

    private void assertGradient(float expected, V value) {
        assertValue(expected, value.gradNode().gradient().orElseThrow());
    }

    /** Checks leaf state and constant exclusion from a seeded backward pass. */
    @Test
    public void explicitSeedsRespectConstants() {
        var x = createValue(2, true);
        var constant = createValue(3, false);
        assertTrue(x.requiresGrad());
        assertFalse(constant.requiresGrad());
        assertTrue(x.valueNode().inputs().isEmpty());
        assertTrue(x.gradNode().gradient().isEmpty());
        assertNull(x.valueNode().backwardFunction());
        var result = x.add(constant);
        assertValue(5, result);
        result.backward(createValue(4, false));
        assertGradient(4, x);
        assertTrue(constant.gradNode().gradient().isEmpty());
        assertValue(2, x);
        assertValue(3, constant);
    }

    /** Checks identity-based traversal after contributions from repeated inputs. */
    @Test
    public void sharedNodesRunOnceAfterAllContributionsArrive() {
        var x = createValue(2, true);
        var shared = x.add(x);
        var result = shared.add(shared);
        var calls = new AtomicInteger();
        var original = shared.valueNode().backwardFunction();
        assertNotNull(original);
        shared.valueNode().setBackwardFunction(value -> {
            calls.incrementAndGet();
            assertGradient(2, value);
            original.accept(value);
        });
        result.backward(createValue(1, false));
        assertEquals(1, calls.get());
        assertGradient(4, x);
    }

    /** Checks accumulation, intermediate replacement and stable gradient handles. */
    @Test
    public void repeatedPassesAndResetPreserveEarlierGradients() {
        var x = createValue(2, true);
        var shared = x.add(x);
        var result = shared.add(shared);
        var handle = x.gradNode();
        result.backward(createValue(1, false));
        var earlier = handle.gradient().orElseThrow();
        assertValue(4, earlier);
        result.backward(createValue(2, false));
        assertGradient(12, x);
        assertGradient(4, shared);
        assertGradient(2, result);
        assertValue(4, earlier);
        handle.reset();
        handle.reset();
        assertTrue(handle.gradient().isEmpty());
        result.backward(createValue(1, false));
        assertSame(handle, x.gradNode());
        assertGradient(4, x);
    }

    /** Checks that leaf entry points accumulate each explicit seed once. */
    @Test
    public void leafBackwardAccumulatesOnlyNewSeeds() {
        var x = createValue(2, true);
        x.backward(createValue(1, false));
        x.backward(createValue(3, false));
        assertGradient(4, x);
    }

    /** Checks failure policies without relying on type-specific implicit seeds. */
    @Test
    public void invalidSeedAndConstantRootPreserveGradientState() {
        var x = createValue(2, true);
        x.backward(createValue(1, false));
        var earlier = x.gradNode().gradient().orElseThrow();
        assertThrows(NullPointerException.class, () -> x.backward(null));
        assertSame(earlier, x.gradNode().gradient().orElseThrow());
        var constant = createValue(3, false);
        assertThrows(IllegalStateException.class,
                () -> constant.backward(createValue(1, false)));
        assertTrue(constant.gradNode().gradient().isEmpty());
    }

    /** Checks leaf tracking changes before constructing each new graph. */
    @Test
    public void trackingChangesOnlyOnLeaves() {
        var x = createValue(2, false);
        var node = x.valueNode();
        var handle = x.gradNode();
        x.setRequiresGrad(true);
        var result = x.add(createValue(3, false));
        assertTrue(result.requiresGrad());
        assertThrows(IllegalStateException.class, () -> result.setRequiresGrad(true));
        result.backward(createValue(1, false));
        x.setRequiresGrad(false);
        assertSame(node, x.valueNode());
        assertSame(handle, x.gradNode());
        assertGradient(1, x);
        assertValue(2, x);
        assertFalse(x.add(createValue(3, false)).requiresGrad());
    }

    /** Checks second derivatives through a retained multiplication graph. */
    @Test
    public void gradientExpressionsRetainHigherDerivatives() {
        var x = createValue(3, true);
        multiply(multiply(x, x), x).backward(createValue(2, false));
        var first = x.gradNode().gradient().orElseThrow();
        assertValue(54, first);
        assertTrue(first.requiresGrad());
        x.gradNode().reset();
        first.backward(createValue(1, false));
        assertGradient(36, x);
        assertValue(54, first);
    }

    /** Checks that replacing a parameter leaves the previous graph independent. */
    @Test
    public void parameterReplacementKeepsOldAndNewLeavesIndependent() {
        var oldLeaf = createValue(2, true);
        var parameter = new ParameterImpl<>(oldLeaf);
        var oldResult = multiply(parameter.value(), parameter.value());
        var newLeaf = createValue(3, true);
        parameter.replace(newLeaf);
        assertSame(newLeaf, parameter.value());
        assertTrue(newLeaf.gradNode().gradient().isEmpty());
        oldResult.backward(createValue(1, false));
        assertGradient(4, oldLeaf);
        assertTrue(newLeaf.gradNode().gradient().isEmpty());
        multiply(parameter.value(), parameter.value()).backward(createValue(1, false));
        assertGradient(6, newLeaf);
        assertGradient(4, oldLeaf);
    }
}
