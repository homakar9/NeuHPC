#!/usr/bin/python3

import random
import typing
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from operator import add
from operator import mul
from operator import pow
from operator import sub
from operator import truediv as div

from functools import reduce

from collections import defaultdict

import seaborn as sns
sns.set()

BINARY_OPS = {
    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div,
    'pow': pow,
}

BINARY_OP_NAMES = {
    add: 'add',
    sub: 'sub',
    mul: 'mul',
    div: 'div',
    pow: 'pow',
}

def reduce_sum(xs):  return reduce(add, xs)
def reduce_prod(xs): return reduce(mul, xs)
def reduce_max(xs):  return reduce(max, xs)

REDUCE_OPS = {
    'reduce_sum': reduce_sum,
    'reduce_prod': reduce_prod,
    'reduce_max': reduce_max,
}

REDUCE_OP_NAMES = {
    reduce_sum: 'reduce_sum',
    reduce_prod: 'reduce_prod',
    reduce_max: 'reduce_max',
}

ACTIVATIONS = {
    'identity': tf.keras.activations.get(None),
    'relu': tf.keras.activations.get('relu'),
    'tanh': tf.keras.activations.get('tanh'),
    'abs': abs,
}


def get_binary_op():
    return random.choice(list(BINARY_OPS))


def get_reduce_op():
    return random.choice(list(REDUCE_OPS))


class Synapse:

    name_suffixes = defaultdict(int)

    def __init__(self, op = 'mul', w = None, name = None):

        if name is None: name = 'synapse'
        suffix = self.name_suffixes[name]
        self.name = f"{name}_{suffix}"
        self.name_suffixes[name] += 1

        # synapse operation
        if op in BINARY_OPS.keys():
            self.op_name = op
            self.op = BINARY_OPS[self.op_name]
        elif op in BINARY_OPS.values():
            self.op_name = BINARY_OP_NAMES[op]
            self.op = op
        else:
            raise ValueError(f"Unrecognized op: {op}")

        # synapse weight
        if w is None:
            w = random.uniform(-0.1, 0.1)

        self.w = tf.Variable(w, name = self.name)


    def __call__(self, activation, trace = False):
        if not trace:
            return self.op(activation, self.w)
        else:
            traced_weight = ('weight', self.w.numpy())
            return self.op(activation, traced_weight)

    def __repr__(self):
        cls = self.__class__.__name__
        w = self.w.numpy()
        return f"{cls}(op = {self.op_name}, w = {w:04f}, name = {self.name})"



class Neuron:

    name_suffixes = defaultdict(int)

    def __init__(self, reduce_op = 'reduce_sum', synapses = None, activation = None, name = None):

        if name is None: name = 'neuron'
        suffix = self.name_suffixes[name]
        self.name = f"{name}_{suffix}"
        self.name_suffixes[name] += 1

        # neuron's reduce operation
        if reduce_op in REDUCE_OPS.keys():
            reduce_op_name = reduce_op
            reduce_op = REDUCE_OPS[reduce_op_name]
        elif reduce_op in REDUCE_OPS.values():
            reduce_op_name = REDUCE_OP_NAMES[reduce_op]
            reduce_op = reduce_op
        else:
            raise ValueError(f"Unrecognized reduce_op: {reduce_op}")

        # neuron's (input) synapses
        if synapses is None:
            synapses = []
        elif isinstance(synapses, typing.Iterable):
            pass
        else:
            raise TypeError(f"Invalid type for argument 'synapses': {synapses}")

        self.reduce_op = reduce_op
        self.reduce_op_name = reduce_op_name
        self.synapses = synapses

        # neuron's activation function
        self.activation_name = activation
        self.activation = ACTIVATIONS[self.activation_name]


    def __repr__(self):
        cls = self.__class__.__name__
        reduce_op = self.reduce_op_name
        num_synapses = len(self.synapses)
        return f"{cls}(reduce_op = {reduce_op}, num_synapses = {num_synapses}, name = {self.name})"

    def __call__(self, xs, trace = False):

        if not isinstance(xs, typing.Iterable):
            raise TypeError(f"Argument to neuron __call__ method must be iterable.")

        if len(xs) != len(self.synapses):
            cls = self.__class__.__name__
            raise ValueError(f"Instance of {cls} has {len(self.synapses)} synapses, "
                             f"but was called with {len(xs)} arguments.")

        dendritic_outputs = [synapse(x, trace=trace) for x, synapse in zip(xs, self.synapses)]

        cell_output = self.reduce_op(dendritic_outputs)

        if trace:
            axon_output = cell_output.activation(self.activation_name) # pass activation *name* to Spies
        else:
            axon_output = self.activation(cell_output)


        return axon_output

    @property
    def variables(self):
        return [synapse.w for synapse in self.synapses]



class Layer:

    def __init__(self, neurons):

        self.neurons = neurons

        input_shapes = [len(neuron.synapses) for neuron in self.neurons]
        input_shape_is_well_defined = (len(set(input_shapes)) == 1)

        if not input_shape_is_well_defined:
            raise ValueError(f"Layer's neurons have differing input shapes: {input_shapes}")

        self.input_shape = input_shapes[0]


    def __repr__(self):
        cls = self.__class__.__name__
        header = f"{cls} with {len(self.neurons)} neurons."
        repr_lines = [header]
        for neuron in self.neurons:
            repr_line = ' * ' + repr(neuron)
            repr_lines.append(repr_line)

        repr_string = '\n'.join(repr_lines) + '\n'
        return repr_string


    def __call__(self, xs, trace = False):
        output = [neuron(xs, trace=trace) for neuron in self.neurons]
        self._output = output # cache for later. we'll need this for the backward pass.
        return output

    @property
    def variables(self):
        return reduce(add, [neuron.variables for neuron in self.neurons])

    @property
    def synapses(self):
        return reduce(add, [neuron.synapses for neuron in self.neurons])


class Network:


    def __init__(self, layers):
        self.layers = layers
        self.input_shape = self.layers[0].input_shape
        self.output_shape = len(self.layers[-1].neurons)


    @classmethod
    def build_from_shapes(cls, num_inputs, list_of_layer_sizes):

        layer_to_num_synapses_per_neuron = dict(enumerate(list_of_layer_sizes))
        layer_to_num_synapses_per_neuron[-1] = num_inputs
        layer_to_num_synapses_per_neuron = dict(sorted(layer_to_num_synapses_per_neuron.items()))

        num_layers = len(list_of_layer_sizes)

        layers = []
        for layer_num, num_neurons in enumerate(list_of_layer_sizes):
            neurons = []
            for neuron_num in range(num_neurons):
                synapses = []
                num_synapses_per_neuron = layer_to_num_synapses_per_neuron[layer_num-1]
                for synapse_num in range(num_synapses_per_neuron):
                    synapse = Synapse(op = get_binary_op())
                    synapses.append(synapse)
                neuron = Neuron(reduce_op = get_reduce_op(), synapses = synapses)
                neurons.append(neuron)
            layer = Layer(neurons)
            layers.append(layer)


        self = cls(layers)
        return self

    @property
    def num_neurons(self):
        return sum([len(layer.neurons) for layer in self.layers])

    @property
    def num_layers(self):
        return len(self.layers)

    def __call__(self, xs, trace=False):

        if not isinstance(xs, typing.Iterable):
            raise TypeError(f"Argument to network __call__ method must be iterable.")

        if len(xs) != self.input_shape:
            cls = self.__class__.__name__
            raise ValueError(f"Instance of {cls} has {self.input_shape} input neurons, "
                             f"but was called with {len(xs)} arguments.")

        activations = {-1: xs} # input_layer will be -1
        for n, layer in enumerate(self.layers):
            activations[n] = layer(activations[n-1], trace=trace)

        N = max(activations.keys())
        return activations[N]


    def __repr__(self):
        cls = self.__class__.__name__
        return (
            f"{cls}("
            f"input_shape = {self.input_shape}, "
            f"output_shape = {self.output_shape}, "
            f"num_layers = {self.num_layers}, "
            f"num_neurons = {self.num_neurons}"
            f")"
        )

    @property
    def variables(self):
        return reduce(add, [layer.variables for layer in self.layers])

    @property
    def synapses(self):
        return reduce(add, [layer.synapses for layer in self.layers])

    @property
    def neurons(self):
        return reduce(add, [layer.neurons for layer in self.layers])


leaky_relu = lambda x: tf.keras.activations.relu(x, alpha = 0.2)

def init_arithmetic_weight(loc = 0.0, scale = 0.1):
    return np.random.normal(loc = loc, scale = scale)

def init_geometric_weight(loc = 1.0, scale = 0.1):
    return np.random.normal(loc = loc, scale = scale)

def make_linear_neuron(num_inputs, name = None):
    synapses = [Synapse(op = 'mul', name = name) for _ in range(num_inputs)]
    neuron = Neuron(reduce_op='reduce_sum', synapses=synapses, activation='identity', name=name)
    return neuron

def make_arithmetic_neuron(num_inputs, activation = 'relu', name = None):
    synapses = [Synapse(op = 'mul', name = name) for _ in range(num_inputs)]
    neuron = Neuron(reduce_op='reduce_sum', synapses=synapses, activation=activation, name=name)
    return neuron

def make_geometric_neuron(num_inputs, activation = 'identity', name = None):
    synapses = [Synapse(op = 'pow', w = init_geometric_weight(), name = name) for _ in range(num_inputs)]
    neuron = Neuron(reduce_op='reduce_prod', synapses=synapses, activation=activation, name=name)
    return neuron


def generate_data(batch_size = 100):
    while True:
        x1 = random.uniform(1.0, 10.0)
        x2 = random.uniform(1.0, 10.0)
        features = (x1, x2)
        label = 1/(x2)
        yield (features, label)


# make network
num_inputs_1 = 2

layer_1 = Layer([
    make_arithmetic_neuron(num_inputs_1, name = 'arithmetic_1'),
    make_arithmetic_neuron(num_inputs_1, name = 'arithmetic_2'),
    make_geometric_neuron(num_inputs_1, name = 'geometric_1'),
])

num_inputs_2 = len(layer_1.neurons)
layer_2 = Layer([
    make_linear_neuron(num_inputs_2, name = 'linear')
])

layers = [layer_1, layer_2]
network = Network(layers)

variable_history = defaultdict(list)

batch_generator = generate_data(batch_size = 1)
num_batches = 1000
learning_rate = 0.01

for batch in range(num_batches):

    features, y_true = next(batch_generator)

    with tf.GradientTape() as tape:

        variables = network.variables

        tape.watch(variables)

        for s in network.synapses:
            variable_history[s.name].append(s.w.numpy())

        [y_pred] = network(features)

        #loss = tf.losses.mean_squared_error(y_true, y_pred)
        loss = 100*(y_true - y_pred)**2

        threshold = 5
        regularizer = tf.reduce_sum(tf.square(variables))
        if regularizer > threshold:
            loss += regularizer

        print(
            f"batch {batch:05d}: "
            f"y_pred was {y_pred:05f}, "
            f"y_true was {y_true:05f}, "
            f"loss was {loss:05f}"
        )

        gradients = tape.gradient(loss, variables)

        for gradient, variable in zip(gradients, variables):

            new_value = variable - (learning_rate * gradient)

            #if ('linear' in variable.name) or ('arithmetic' in variable.name):
            #    new_value = variable - (learning_rate * gradient)
            #elif ('geometric' in variable.name):
            #    new_value = variable - (learning_rate * gradient)
            #else:
            #    raise ValueError(f"Neuron type could not be determined from name: '{variable.name}'")

            variable.assign(new_value)


###################################################################
### Now we'll simplify the network to make it more parsimonious ###
###################################################################

import tensorwoah as tw
#x = tw.Variable(value = tw.Spy('x'), name = 'x')
#y = x**5
#grad = y.gradients()

spy_names = ['x', 'y']
spies = [tw.Spy(name = spy_name) for spy_name in spy_names]
[spy_report] = network(spies, trace=True)

# generate symbolic expression from spy report
import sympy
from sympy import Piecewise

SympyRelu = lambda x: Piecewise((x, x > 0), (0, x <= 0))

spy_map = {spy_name : sympy.Symbol(spy_name) for spy_name in spy_names}


def parse_espionage(expression, eps=0.05):

    # composite terms
    op_type, *args = expression

    if op_type == 'identity':
        return parse_espionage(args[0], eps=eps)
    elif op_type == 'add':
        return reduce(add, [parse_espionage(arg, eps=eps) for arg in args])
    elif op_type == 'mul':
        return reduce(mul, [parse_espionage(arg, eps=eps) for arg in args])
    elif op_type == 'pow':
        assert len(args) == 2, f"oops pow: {args}"
        return pow(*[parse_espionage(arg, eps=eps) for arg in args])
    elif op_type == 'weight':
        assert len(args) == 1, f"oops weight: {args}"
        weight = args[0]
        if abs(weight - round(weight)) < eps:
            weight = int(round(weight))
        return weight
    elif op_type == 'relu':
        #return SympyRelu(parse_espionage(args[0], eps=eps))
        return parse_espionage(args[0], eps=eps)

    # atomic terms
    elif expression.history is None and expression.name in spy_map:
        term = spy_map[expression.name]
        return term

    else:
        raise NotImplementedError(f"Not sure what to do with expression: {expression}")


newton_outputs = {}
for eps in (0.20, 0.10, 0.05, 0.01, 0.005, 0.001):
    newton_output = parse_espionage(spy_report, eps=eps).simplify()
    newton_outputs[eps] = newton_output
    print(f"At weight tolerance {eps}, network learned symbolic expression: {newton_output}")


### NOW PLOT IT
plt.ioff()

fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')

synapse_names = sorted(list(variable_history.keys()))

for i, j in itertools.product(range(3), range(3)):
    index = 3*i + j
    synapse_name = synapse_names[index]
    series = variable_history[synapse_name]
    final_value = series[-1]
    final_value = round(final_value, 2)
    #ax[i,j].set_ylim(-2,+2)
    ax[i,j].plot(series)
    ax[i,j].set_title(f"{synapse_name}: {final_value:.02f}")


symbolic_expression = newton_outputs[0.05]

plt.suptitle(
    f"Individual weights over time for hybrid network.\n"
    f"Newton Learned $f(x, y) = {symbolic_expression}$"
)
plt.show()

