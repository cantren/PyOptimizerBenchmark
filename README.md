# PyOptimizerBenchmark
A simple python class to evaluate and compare numeric optimizer performance.

# PyOptimizerBenchmark Quick-Start Guides

This README provides quick-start guides for using the PyOptimizerBenchmark with various optimization libraries.

---

## PyTorch

```markdown
### PyOptimizerBenchmark with PyTorch

PyTorch provides tools for automatic differentiation and optimization. This guide demonstrates using PyOptimizerBenchmark with PyTorch for gradient-based optimization.

#### Getting Started

Install PyTorch using pip:

```bash
pip install torch
```

#### Example Usage

```python
import torch
from torch.optim import Adam
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()

def rosenbrock_torch(x, y):
    x_tensor = torch.tensor([x], requires_grad=True)
    y_tensor = torch.tensor([y], requires_grad=True)
    
    value = benchmark.rosenbrock(x_tensor, y_tensor)
    value.backward()
    
    return value.item(), x_tensor.grad.item(), y_tensor.grad.item()

x, y = 0.0, 0.0  # Starting point
optimizer = Adam([torch.tensor(x), torch.tensor(y)], lr=0.01)

for _ in range(1000):
    optimizer.zero_grad()
    value, grad_x, grad_y = rosenbrock_torch(x, y)
    optimizer.step()
    
    x, y = optimizer.param_groups[0]['params']
    print(f"Value: {value}, Gradient: ({grad_x}, {grad_y})")

print(f"Optimized x: {x}, y: {y}")
```
```

---

## TensorFlow 2

```markdown
### PyOptimizerBenchmark with TensorFlow 2

TensorFlow 2's optimization capabilities are used in this guide to integrate PyOptimizerBenchmark for complex tasks.

#### Getting Started

Install TensorFlow 2:

```bash
pip install tensorflow
```

#### Example Usage

```python
import tensorflow as tf
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()

x = tf.Variable(0.0, dtype=tf.float32)
y = tf.Variable(0.0, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.01)

for _ in range(1000):
    with tf.GradientTape() as tape:
        value = benchmark.rosenbrock(x, y)
    gradients = tape.gradient(value, [x, y])
    optimizer.apply_gradients(zip(gradients, [x, y]))
    
    print(f"Value: {value.numpy()}, Gradients: {[g.numpy() for g in gradients]}")

print(f"Optimized x: {x.numpy()}, y: {y.numpy()}")
```
```

---

## GLOP

```markdown
### PyOptimizerBenchmark with GLOP

GLOP, part of Google OR-Tools, is used for linear approximations of PyOptimizerBenchmark functions.

#### Getting Started

Install Google OR-Tools:

```bash
pip install ortools
```

#### Example Usage

```python
from ortools.linear_solver import pywraplp
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()
solver = pywraplp.Solver.CreateSolver('GLOP')

x = solver.NumVar(-2.0, 2.0, 'x')
y = solver.NumVar(-2.0, 2.0, 'y')

f_0 = benchmark.rosenbrock(1, 1)
grad_x_0, grad_y_0 = benchmark.rosenbrock_gradient(1, 1)
linear_objective = f_0 + grad_x_0 * (x - 1) + grad_y_0 * (y - 1)
solver.Minimize(linear_objective)

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())
else:
    print('The problem does not have an optimal solution.')
```
```

---

## PuLP

```markdown
### PyOptimizerBenchmark with PuLP

This guide shows how to use PuLP for linear programming approximations of PyOptimizerBenchmark functions.

#### Getting Started

Install PuLP:

```bash
pip install pulp
```

#### Example Usage

```python
import pulp
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()
prob = pulp.LpProblem("MinimizeFunction", pulp.LpMinimize)

x = pulp.LpVariable("x", -5, 5)
y = pulp.LpVariable("y", -5, 5)

f_0 = benchmark.rosenbrock(1, 1)
grad_x_0, grad_y_0 = benchmark.rosenbrock_gradient(1, 1)
prob += f_0 + grad_x_0 * (x - 1) + grad_y_0 * (y - 1)

prob.solve()
print("Status:", pulp.LpStatus[prob.status])
print("Optimal value:", pulp.value(prob.objective))
print("Optimal x:", x.varValue)
print("Optimal y:", y.varValue)
```
```

---

## Pyomo

```markdown
### PyOptimizerBenchmark with Pyomo

Pyomo is utilized for formulating and solving both linear and nonlinear optimization problems with PyOptimizerBenchmark.

#### Getting Started

Install Pyomo:

```bash
pip install pyomo
```

#### Example Usage

```python
import pyomo.environ as pyo
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()
model = pyo.ConcreteModel()

model.x = pyo.Var(bounds=(-5, 5))
model.y = pyo.Var(bounds=(-5, 5))

def rosenbrock_rule(model):
    return benchmark.rosenbrock(model.x(), model.y())
model.obj = pyo.Objective(rule=rosenbrock_rule, sense=pyo.minimize)

solver = pyo.SolverFactory('ipopt')
results = solver.solve(model)

print("Status:", results.solver.status)
print("Optimal value:", pyo.value(model.obj))
print("Optimal x:", pyo.value(model.x))
print("Optimal y:", pyo.value(model.y))
```
```

---

## CVXPY

```markdown
### PyOptimizerBenchmark with CVXPY

CVXPY is used for convex optimization tasks. This guide demonstrates integration with PyOptimizerBenchmark.

#### Getting Started

Install CVXPY:

```bash
pip install cvxpy
```

#### Example Usage

```python
import cvxpy as cp
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()
x = cp.Variable()
y = cp.Variable()

objective = cp.Minimize(cp.square(1 - x) + 100 * cp.square(y - cp.square(x)))
constraints = [cp.square(x) + cp.square(y) <= 1]
prob = cp.Problem(objective, constraints)

prob.solve()
print("Status:", prob.status)
print("Optimal value:", prob.value)
print("Optimal x:", x.value)
print("Optimal y:", y.value)
```
```

---

## CPLEX

```markdown
### PyOptimizerBenchmark with CPLEX

CPLEX, known for solving linear and quadratic problems, is used with PyOptimizerBenchmark in this guide.

#### Getting Started

Ensure CPLEX is installed and accessible via Python.

#### Example Usage

```python
import cplex
from PyOptimizerBenchmark import PyOptimizerBenchmark

benchmark = PyOptimizerBenchmark()
cpx = cplex.Cplex()

cpx.variables.add(names=["x", "y"], lb=[-5]*2, ub=[5]*2)

# Linear approximation of Rosenbrock
cpx.objective.set_sense(cpx.objective.sense.minimize)
cpx.objective.set_linear([("x", -2 * (1 - 1) - 400 * (1 - 1**2)), ("y", 200 * (1 - 1**2))])

cpx.solve()

solution_status = cpx.solution.get_status()
print("Solution status:", cpx.solution.status[solution_status])
if solution_status == cpx.solution.status.optimal:
    print("Optimal value:", cpx.solution.get_objective_value())
    print("Values: x =", cpx.solution.get_values("x"), "y =", cpx.solution.get_values("y"))
else:
    print("No optimal solution found.")
```
```
