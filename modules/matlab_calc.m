function [lower_bounds_enDive,upper_bounds_enDive, lower_inf, upper_inf] = matlab_calc(data, kernel)

    lower_bounds_enDive = zeros(1, length(data));
    upper_bounds_enDive = zeros(1, length(data));
    upper_inf = zeros(1, length(data));
    lower_inf = zeros(1, length(data));

    % Parfor loop to parallelize calculations
    % size(data)
    % length(data)
    parfor i = 1:size(data, 1)    
        sim_i = squeeze(data(i, :, :, :)); % Use curly braces if 'data' is a cell array
        
        data0 = squeeze(sim_i(1, : , :));  % Correct indexing for cell array
        data1 = squeeze(sim_i(2, : , :));  % Correct indexing for cell array


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
