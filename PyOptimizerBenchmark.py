class PyOptimizerBenchmark:
    def __init__(self):
        # Initialize any required variables or constants
        self.r_T = 1
        self.r_S = 0.2
        self.n = 8

    # Rosenbrock function and its gradient with optional constraints
    def rosenbrock(self, x, y, constrained=0):
        value = (1 - x)**2 + 100 * (y - x**2)**2
        if constrained == 1 and not (((x - 1)**3 - y + 1 <= 0) and (x + y - 2 <= 0)):
            return np.nan
        if constrained == 2 and not (x**2 + y**2 <= 2):
            return np.nan
        return value

    def rosenbrock_gradient(self, x, y):
        grad_x = -2 * (1 - x) - 4 * 100 * x * (y - x**2)
        grad_y = 2 * 100 * (y - x**2)
        return np.array([grad_x, grad_y])

    # Mishra's Bird function and its gradient with optional constraint
    def mishras_bird(self, x, y, constrained=False):
        sin_y = np.sin(y)
        exp_term = np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi))
        value = np.sin(x) * exp_term + (x - y)**2 - 100*(sin_y)**2 - (x/2 - y - 1)**4
        if constrained and ((x + 5)**2 + (y + 5)**2 >= 25):
            return np.nan
        return value

    def mishras_bird_gradient(self, x, y):
        r = np.sqrt(x**2 + y**2)
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        sin_y = np.sin(y)
        cos_y = np.cos(y)
        exp_term = np.exp(np.abs(1 - r / np.pi))

        grad_x = cos_x * exp_term + 2 * (x - y) - 4 * (x/2 - y - 1)**3
        grad_y = -exp_term * sin_x * y / (r * np.pi) * np.sign(1 - r / np.pi) + \
                 2 * (y - x) - 200 * sin_y * cos_y - 4 * (x/2 - y - 1)**3

        return np.array([grad_x, grad_y])

    # Townsend function and its gradient with optional constraint
    def townsend(self, x, y, constrained=False):
        term1 = -np.cos((x - 0.1) * y)**2
        term2 = -x * np.sin(3 * x + y)
        value = term1 + term2
        if constrained:
            t = np.arctan2(y, x)
            r_squared = (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t))**2 + (2 * np.sin(t))**2
            if x**2 + y**2 >= r_squared:
                return np.nan
        return value

    def townsend_gradient(self, x, y):
        cos_term = np.cos((x - 0.1) * y)
        sin_term = np.sin((x - 0.1) * y)
        sin_term2 = np.sin(3 * x + y)

        grad_x = 2 * y * cos_term * sin_term + (-np.sin(3 * x + y) - 3 * x * np.cos(3 * x + y))
        grad_y = (x - 0.1) * 2 * cos_term * sin_term - x * np.cos(3 * x + y)

        return np.array([grad_x, grad_y])

    # Gomez and Levy function and its gradient with optional constraint
    def gomez_levy(self, x, y, constrained=False):
        value = 4*x**2 - 2.1*x**4 + x**6 / 3 + x*y - 4*y**2 + 4*y**4
        if constrained and not (-np.sin(4*np.pi*x) + 2*np.sin(2*np.pi*y)**2 <= 1.5):
            return np.nan
        return value

    def gomez_levy_gradient(self, x, y):
        grad_x = 8*x - 8.4*x**3 + 2*x**5 + y
        grad_y = x - 8*y + 16*y**3
        return np.array([grad_x, grad_y])

    # Simionescu function and its gradient with optional constraint
    def simionescu(self, x, y, constrained=False):
        value = 0.1 * x * y
        if constrained:
            t = np.arctan2(y, x)
            domain_constraint = (self.r_T + self.r_S * np.cos(self.n * t))**2
            if x**2 + y**2 >= domain_constraint:
                return np.nan
        return value

    def simionescu_gradient(self, x, y):
        grad_x = 0.1 * y
        grad_y = 0.1 * x
        return np.array([grad_x, grad_y])
