import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

symbolic = True
neumann_conditions = True

# Compute error on cubic splines for range of number of points to interpolate. Must be fixed, cannot be a symbolic parameter
# n_pts E [3 , 5 , 7 , 9]
for n_pts in range(5, 6, 2):

    # Distance between two points /!\ Hypothesis : h is constant,
    # might want to change that later but it represents the distance between two voxels, which is constant.
    h = sp.symbols('h') if symbolic else 0.5

    # Offset of the domain of interpolation compared to the function
    xi = sp.symbols('xi') if symbolic else 0

    # Slope factor for density function
    k = sp.symbols('k') if symbolic else 4


    # symbolic parameters
    dom_len = h*(n_pts-1)    # Length of the domain on which to interpolate
    h0 = -dom_len/2 + xi     # Position of the first point to interpolate   
        
    # Spline coefficients
    eqns = []
    a = [sp.symbols('a%d' % i) for i in range(1, n_pts + 1)]
    b = [sp.symbols('b%d' % i) for i in range(1, n_pts + 1)]
    c = [sp.symbols('c%d' % i) for i in range(1, n_pts + 1)]
    d = [sp.symbols('d%d' % i) for i in range(1, n_pts + 1)]

    # Compute point x value on axis based on point index
    def compute_point_x(index):
        return h0 + index*h

    # Define gradient function
    def grad(x):
        if symbolic:
            #return k*x
            #return -k*x**2
            return k*(1 - sp.tanh(k*x)**2)
        else :
            #return x
            #return -x**2
            #return x**3
            return k*(1 - np.tanh(k*x)**2)
        
    # Print the cublic splines in console
    def print_splines():
        for i in range(0, n_pts-1):
            x_i = compute_point_x(i)
            spline_string = f"y{i} = {a[i]} + ({solution[b[i]]})(x-({x_i})) + ({solution[c[i]]})(x-({x_i}))² + ({solution[d[i]]})(x-({x_i}))³"
            print(spline_string)


    # /!\ Indexing is not shifted relative to the paper's indexing, so b1 in the paper stays b1 here, but it is stored in b[0] !

    # First fill the a values
    for i in range(0, n_pts):
        a[i] = grad(compute_point_x(i))

    # Create the system of linear equations
    for i in range(0, n_pts-1):
        if i != 0:
            eqns.append( (1/3)*h*c[i-1] + (2/3)*2*h*c[i] + (1/3)*h*c[i+1]
                        - (grad(compute_point_x(i+1)) - 2*grad(compute_point_x(i)) + grad(compute_point_x(i-1)))/h )

        eqns.append( (c[i+1]-c[i])/(3*h) - d[i] )
        eqns.append( (grad(compute_point_x(i+1)) - grad(compute_point_x(i)))/h - (2/3)*h*c[i] - (1/3)*h*c[i+1] - b[i] )
        
    # Add boundary conditions
    if neumann_conditions:
        eqns.append( b[0] )
        eqns.append( b[n_pts-1] )
    else:
        eqns.append( c[0] )
        eqns.append( c[n_pts-1] )

    # Quadratic extrapolation on boundary
    eqns.append( d[n_pts-1] )
    eqns.append( 3*d[n_pts-2]*h**2 + 2*c[n_pts-2]*h + b[n_pts-2] - b[n_pts-1] )

    # Solve the system
    solution = sp.solve(eqns, b+c+d)

    # Compute the derivative of the splines to find the maximum
    if symbolic:
        derivative_solutions = []
        x = sp.symbols('x')
        for i in range(0, n_pts-1):
            x_i = compute_point_x(i)
            expr = a[i] + solution[b[i]]*(x-x_i) + solution[c[i]]*(x-x_i)**2 + solution[d[i]]*(x-x_i)**3
            dx = sp.diff(expr, x)
            #print(f"x value of maximum for spline {i} : ")
            #print(sp.solve(derivative, x))
            derivative_solutions.append(sp.solve(dx, x))

        # Evaluate derivative solutions for range of parameter values, and plot the result.
        # /!\ on the plot we should have in x the distance between the surface and the closest voxel instead of x_i which does not represent exactly that
        # Choose values for h, k
        derivative_solutions_evaluations = []
        k_over_h_ratio = 8 # Revise this ratio
        k_param = 4
        h_param = k_param/k_over_h_ratio

        #xi_range = [x*h_param/20 for x in range(-80, 80)]
        xi_range = [x*h_param/20 for x in range(0, 1)]
        for xi_param in xi_range:
            max_expr_value = 0
            max_root = np.nan
            for i in range(0, n_pts-1):
                print(f"{i} with xi = {xi_param} for h = {h_param}")
                print(derivative_solutions[i])
                for j in range(0, 2):
                    root = derivative_solutions[i][j].subs([(k, k_param), (h, h_param), (xi, xi_param)])
                    print(f"i={i}, root {j} : {root}")
                    # Must exclude roots which are not within the domain of the spline
                    h0_param = h0.subs([(h, h_param), (xi, xi_param)])
                    if(sp.im(root) != 0 or root < h0_param + i*h_param or root > h0_param + (i+1)*h_param):
                        continue

                    val = expr.subs([(k, k_param), (h, h_param), (xi, xi_param), (x, root)])
                    print(f"")
                    print(f"max_expr_value : {max_expr_value}, expr_value : {val}")
                    if abs(val) > abs(max_expr_value):
                        max_expr_value = val
                        max_root = root
            
            print(max_root)
            derivative_solutions_evaluations.append(max_root)
        # I will pretend that xi is the distance between the surface and the closest voxel, but it is false in the case of an even number of points
        plt.plot(xi_range, derivative_solutions_evaluations)

if symbolic:
    plt.legend(["nb_points = 3", "nb_points = 5", "nb_points = 7", "nb_points = 9"])
    for i in range(int(-n_pts/2), int((n_pts+1)/2)):
        plt.axvline(x=i*h_param, color='red', linestyle='--')
    plt.grid()
    plt.xlabel('xi (Distance between surface and central voxel)')
    plt.ylabel('Error on the interpolation')
    plt.title(f"Error on cubic spline interpolation (k={k_param}, h={h_param}) as a function of xi (the distance between surface and central voxel)")
    plt.show()





# Print the resulting spline equations
#print_splines();

# Plot resulting splines
if not symbolic:
    plot_step = 0.001
    xs = np.arange(h0, h0 + dom_len, plot_step)     # Plot domain
    #ts = np.arange(h0, h0 + dom_len, plot_step/100) # Test domain
    ys = np.zeros(xs.shape)
    dx = np.zeros(xs.shape)

    for i in range(0, n_pts-1):
        A = a[i]
        B = solution[b[i]]
        C = solution[c[i]]
        D = solution[d[i]]

        x0 = compute_point_x(i)
        interp = lambda x: A + B*(x-x0) + C*(x-x0)**2 + D*(x-x0)**3
        deriv = lambda x: B + 2*C*(x-x0) + 3*D*(x-x0)**2

        for x_i in range(int(i*len(xs)/(n_pts-1)), int((i+1)*len(xs)/(n_pts-1))):
            ys[x_i] = interp(xs[x_i])
            dx[x_i] = deriv(xs[x_i])
            

    f = grad(xs)

    plt.plot(xs, ys)
    plt.plot(xs, f, color='red')
    plt.plot(xs, dx, color='green')
    plt.legend(["Splines", "G(x)", "dG(x)/dx"])

    max_index = np.argmax(ys)
    max_x_value = xs[max_index]
    plt.axvline(x=max_x_value, color='green', linestyle='--')
    plt.text(max_x_value, ys[max_index], f'{max_x_value}', color='green', verticalalignment='bottom')
    plt.title(f"Cubic spline interpolation (nb_points={n_pts}, k={k}, h={h}, xi={xi})")
    #plt.xticks(np.arange(-dom_len/2, dom_len/2, step=0.5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()