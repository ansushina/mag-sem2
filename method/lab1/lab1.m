function lab1()

    global calls;
    calls = 0;
    a = 0;
    b = 1;
    eps = power(10, -6);
    delta = (b-a)/ 4;
    x0 = a;
    f0 = f(x0);
    
    N = 0;
        
    while 1
        x1 = x0 + delta;
        f1= f(x1);
        N = N+1;
        fprintf('#%d - x: %20.15f, f(x): %20.15f\n', N, x1, f1);
        if (f0 > f1) 
           x0 = x1;
           f0 = f1;
           if (x0 > a && x0 < b)
                continue;
           end
        end
        
        if (abs(delta) <= eps) 
            break
        else 
            x0=x1; 
            f0=f1;
            delta = - delta / 4;
        end
    end
        
 
    xres = x0;
    fres = f0;
    
    print_result(a, b , eps, xres, fres);
    
    x = a: 1e-2: b;
    fx = f(x);
    plot(x, fx);
end

function y = f(x)
   global calls;
    %y = tan( (2*power(x, 4)- 5*x + 6) ./ 8) + atan((7*power(x, 2) - 11*x + 1 - sqrt(2))./ (-7*power(x, 2) + 11*x + sqrt(2)));
    %y = (1-cos(x-0.777)).^4;
    y = power(x - 0.1, 2) + 0.6;
    calls = calls+1;
end

function print_result(a, b, eps, x, fres)
        global calls;
   
   
        fprintf('[%d, %d]; eps=%20.15f; x = %20.15f ; f(x): %20.15f\n', a, b, eps, x, fres);
        
        fprintf('Calls count: %d\n', calls);

end