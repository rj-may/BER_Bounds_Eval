function [lower_bounds_enDive,upper_bounds_enDive, lower_inf, upper_inf] = matlab_calc(data_set, kernel)

    mc_iter = size(data_set, 1);

    lower_bounds_enDive = zeros(1, mc_iter);
    upper_bounds_enDive = zeros(1, mc_iter);
    upper_inf = zeros(1, mc_iter);
    lower_inf = zeros(1, mc_iter);

    % Parfor loop to parallelize calculations
    % size(data)
    % length(data)
    dim3 = size(data_set, 3);
    dim4 = size(data_set, 4);



    parfor i = 1:mc_iter
        
        % data0 = data_set(i, 1, : , :);  
        % data1 = data_set(i, 2, : , :); 

        data0 = reshape(data_set(i, 1, : , :), [dim3, dim4]);
        data1 = reshape(data_set(i, 2, : , :), [dim3, dim4]);

        % Calculate EnDive
        Dp = EnDive(data0, data1, 'type', "DP", 'quiet', 'kernel', kernel, 'est', 2, nargout=1);

        if Dp > 1   % This is the "ghetto" fix you mentioned, clamping Dp to 1
            Dp  = 1;
        end

        if Dp > 0
            u = 1/2 - 1/2 * Dp;  % Calculate upper bound
            l = 1/2 - 1/2 * sqrt(Dp);  % Calculate lower bound
        else
            l = 1/2;
            u = 1/2;  % When Dp is 0 or negative, set both bounds to 1/2
        end

        lower_bounds_enDive(i) = l;
        upper_bounds_enDive(i) = u;


        
        % Calculate Hellinger divergence
        hellinger_dist = hellingerDivergence(data0, data1, [], []);
        BC = 1 - hellinger_dist;
        up = 1/2 * BC ;
        low = 1/2 - 1/2 * sqrt(1- BC^2);      
        lower_inf(i) = low;
        upper_inf(i) = up;


    end
end
