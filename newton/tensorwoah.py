#!/usr/bin/env python3


class Variable:

    """
        Defining a learnable variable.

        This class uses programmable syntax to define a special type of 
        variable that can observe and track the computations performed on 
        itself. Such a variable can compute derivatives, with respect
        to itself, of computations into which it enters.

        In doing so, a variable of this form gains automatic knowledge
        of *how to change itself* in order to improve computed metrics
        of quality.

        For example, when 2*x is computed, the variable x on which this
        operation is performed will return the result as another instance
        of the same class, adding itself to the set of dependencies of the
        resulting object so that the result will know the variables
        (and computations) on which it depends, as well as their derivatives. 
        This will allow computed measures of quality to be automatically
        improved, by allowing the variables on which they depend to modify
        themselves in the direction of increasing quality.

        Tl;dr: Reimplementing Tensorflow from scratch, no imports.
    """

    _add_ops = []
    _mul_ops = []
    _pow_ops = []
    _const_ops = []

    def __init__(
            self,
            value = None,
            deps = None,
            op = None,
            name = None,
            is_intermediate = False,
        ):
        self.value = value
        self.type = type(value)
        self.name = name
        self.deps = [] if deps is None else deps
        self.is_intermediate = is_intermediate

        if op is None:
            self.op = ('constant', self.value)
            self.direct_deps = []
        else:
            op_name, *variables = op
            assert isinstance(op_name, str)
            for variable in variables:
                assert isinstance(variable, Variable)
            self.op = op
            self.op_name = op_name
            self.direct_deps = variables


    def __mul__(self, other):

        """
            Occurs when `self * other` is performed.
        """

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('mul', self, other)

        cls = type(self)
        retval = self.value * other.value
        deps = cls.getdeps(self, other)
        op_name = self._register_mul_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)


    def __rmul__(self, other):

        """
            Occurs when `other * self` is performed,
            and the `other` object does not know how to
            multiply itself by objects of type `type(self)`
        """

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('rmul', other, self)

        cls = type(self)
        retval = other.value * self.value
        deps = cls.getdeps(self, other)
        op_name = self._register_mul_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)


    def __add__(self, other):

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('add', self, other)

        cls = type(self)
        retval = self.value + other.value
        deps = cls.getdeps(self, other)
        op_name = self._register_add_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)


    def __radd__(self, other):

        """
            Occurs when `other + self` is performed,
            and the `other` object does not know how to
            add itself to objects of type `type(self)`
        """

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('add', other, self)

        cls = type(self)
        retval = other.value + self.value
        deps = cls.getdeps(self, other)
        op_name = self._register_add_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)

    def __pow__(self, other):

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('pow', self, other)

        cls = type(self)
        retval = self.value ** other.value
        deps = cls.getdeps(self, other)
        op_name = self._register_pow_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)


    def __rpow__(self, other):

        """
            Occurs when `other ** self` is performed,
            and the `other` object does not know how to
            raise itself to objects of type `type(self)`
        """

        if not isinstance(other, Variable):
            name = self._register_const_op_and_get_default_name()
            other = Variable(other, name = name, is_intermediate = True)

        op = ('pow', other, self)

        cls = type(self)
        retval = other.value ** self.value
        deps = cls.getdeps(self, other)
        op_name = self._register_pow_op_and_get_default_name()
        return cls(retval, deps = deps, name = op_name,
                    op = op, is_intermediate = True)


    def __eq__(self, other):
        other_value = other.value if isinstance(other, Variable) else other
        return self.value == other_value


    def __hash__(self):
        return id(self)


    def __repr__(self):
        cls = self.__class__.__name__
        args = [self.value]
        if self.name is not None:
            args.append(f'name = {self.name}')
        arglist = ', '.join(map(str, args))
        return f"{cls}({arglist})"


    def _register_const_op_and_get_default_name(self):
        cls = type(self)
        num_const_ops = len(cls._const_ops)
        name = f'constant:{num_const_ops}'
        cls._const_ops.append(self)
        return name


    def _register_add_op_and_get_default_name(self):
        cls = type(self)
        num_add_ops = len(cls._add_ops)
        name = f'add:{num_add_ops}'
        cls._add_ops.append(self)
        return name


    def _register_mul_op_and_get_default_name(self):
        cls = type(self)
        num_mul_ops = len(cls._mul_ops)
        name = f'mul:{num_mul_ops}'
        cls._mul_ops.append(self)
        return name


    def _register_pow_op_and_get_default_name(self):
        cls = type(self)
        num_pow_ops = len(cls._pow_ops)
        name = f'pow:{num_pow_ops}'
        cls._pow_ops.append(self)
        return name


    @classmethod
    def getdeps(cls, *objs):
        deps = []
        objs = [*set(objs)]
        deps += objs
        for obj in objs:
            try:
                deps += [o for o in obj.deps]
            except AttributeError:
                pass
        return [o for o in set(deps) if not o.is_intermediate]


    @classmethod
    def compute_gradient(cls, output_var, input_var):

        gradient = cls.compute_gradient

        op_name, *args = op = output_var.op

        op_type = op_name.split(':')[0]

        if output_var == input_var:
            return 1.0

        elif op_type in {'mul', 'rmul'}:
            arg1, arg2 = args
            return arg1*gradient(arg2, input_var) \
                 + arg2*gradient(arg1, input_var)

        elif op_type in {'add', 'radd'}:
            arg1, arg2 = args
            return gradient(arg1, input_var) \
                 + gradient(arg2, input_var)

        elif op_type in {'pow', 'rpow'}:
            arg1, arg2 = args
            return arg2 * (arg1 ** (arg2 + (-1))) * gradient(arg2, input_var)
            ## need this in general, but log not yet supported
            #return output_var * (
            #     gradient(arg2, input_var) * log(arg1)
            #     + (arg2 / arg1) * gradient(arg1, input_var)
            #)

        elif op_type in {'constant'}:
            return 0.0

        else:
            raise ValueError(f"Gradient not supported for op_type {op_type}")


    def gradient(self, input_var):
        cls = type(self)
        return cls.compute_gradient(self, input_var)

    def gradients(self):
        return {dep : self.gradient(dep) for dep in self.deps}


class Spy:

    def __init__(self, history = None, name = None):

        if history is None and name is None:
            raise ValueError(f"One of history or name must be specified.")

        self.history = history
        self.name = name

    def __add__(self, other):
        return Spy(('add', self, other))

    def __radd__(self, other):
        return Spy(('add', other, self))

    def __mul__(self, other):
        return Spy(('mul', self, other))

    def __rmul__(self, other):
        return Spy(('mul', other, self))

    def __pow__(self, other):
        return Spy(('pow', self, other))

    def __rpow__(self, other):
        return Spy(('pow', other, self))

    def __repr__(self):
        if self.history is not None:
            return repr(self.history)
        else:
            return str(self.name)

    def __iter__(self):
        if self.history is not None:
            return iter(self.history)
        else:
            return iter([self.name])

    def activation(self, activation_name):
        return Spy((activation_name, self))


def example_usage():

    a = Variable(3.0, name = 'a')
    b = Variable(4.0, name = 'b')
    c = a*a
    d = a*a*a
    e = a*b*a
    f = a*(3*(a*b) + 1)

    assert c.gradient(a) == 6
    assert d.gradient(a) == 27


    # now verify that the gradient method works continuously

    import numpy as np
    import matplotlib.pyplot as plt
    from functools import reduce

    xs = np.linspace(-2.0, +2.0, 100)
    y1s, y2s, y3s = [], [], []

    x = Variable(0.0)

    for xvalue in xs:

        # x = Variable(x)   # was originally doing this `for x in xs`
        x.value = xvalue    # don't need to create a new variable every time.

        y1 = x*x

        # now let's differentiate a recursive computation.
        def reduce_multiply(xs, acc = 1.0):
            return reduce_multiply(xs[1:], acc = acc * xs[0]) \
            if xs else acc

        y2 = reduce_multiply([x, x, x])

        # now let's differentiate a for loop.
        y3 = 1.0
        for n in range(4):
            y3 = x*y3

        y1s.append(y1.gradient(x).value)
        y2s.append(y2.gradient(x).value)
        y3s.append(y3.gradient(x).value)

    plt.plot(xs, y1s, label = f"Derivative of $x^2$.")
    plt.plot(xs, y2s, label = f"Derivative of recursive computation of $x^3$.")
    plt.plot(xs, y3s, label = f"Derivative of for loop computation of $x^4$.")

    plt.title(
        f"Derivatives of programs, computed by variables\n"
        f"that can observe the operations performed on them."
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example_usage()

