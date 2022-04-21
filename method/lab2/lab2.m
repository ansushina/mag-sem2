function lab2()
    global calls;
    calls = 0;
    astart = 0;
    bstart = 1;
    eps = power(10, -6);
    N = 0;
    
    a = astart;
    b = bstart;
    
    tau = (sqrt(5) - 1)/2;
    l = b-a;
    
    x1 = b - tau*l;
    x2 = a+ tau*l;
    
    f1 = f(x1);
    f2 = f(x2); 
    
    while (l > 2*eps) 
        N = N+1;
        if (f1 <= f2)
            b = x2;
            l = b-a;
            
            x2=x1;
            f2=f1;
            
            x1 = b - tau*l;
            f1 = f(x1);
        else
            a = x1;
            l = b - a;
            
            x1 = x2;
            f1 = f2;
            
            x2 = a + tau * l;
            f2 = f(x2);
        end
        
        fprintf('#%d -  [ %20.15f , %20.15f ]\n', N, a, b);
    end
 
 
    xres = (a+b)/2;
    fres = f(xres);
    
    print_result(astart, bstart , eps, xres, fres);
    
    x = astart: 1e-2: bstart;
    fx = f(x);
    plot(x, fx);
end

function y = f(x)
   global calls;
    y = tan( (2*power(x, 4)- 5*x + 6) ./ 8) + atan((7*power(x, 2) - 11*x + 1 - sqrt(2))./ (-7*power(x, 2) + 11*x + sqrt(2)));
    calls = calls+1;
end

function print_result(a, b, eps, x, fres)
    global calls;
    fprintf('[%d, %d]; eps=%20.15f; x = %20.15f ; f(x): %20.15f\n', a, b, eps, x, fres);     
    fprintf('Calls count: %d\n', calls);
end