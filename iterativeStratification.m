% ITERATIVESTRATIFICATION Create cross-validation partition for data.
% 
%   C = ITERATIVESTRATIFICATION(Y, N) partitions a multi-label Y into N
%   folds. Y is a matrix of binary labels represented by numbers {0, 1}. 
%   Each fold in the result vector C will have approximately the same
%   distribution of positive and negative samples of each label.
% 
% Limitations:
%   The algorithm is greedy. Hence, the solution is only approximate.
%  
% Example:
%   y = randi(2, 16, 3) - 1             % 3 binary labels
%   n = 5                               % partition into 5 folds
%   rnd(2001)                           % for reproducibility
%   cv = iterativeStratification(y, n)
% 
% Reference: 
%   On the Stratification of Multi-label Data by Sechidis, Konstantinos
%   & Tsoumakas, Grigorios & Vlahavas, Ioannis.
%
% See also CVPARTITION, CROSSVAL.
function solution = iterativeStratification(y, n)
% Argument validation
validateattributes(y, {'numeric'}, {'2d'})
validateattributes(n, {'numeric'}, {'scalar', 'positive', 'nonnan', 'finite'})
assert(round(n) == n, 'The count of folds must be a whole number')
assert(max(y(:))<=1, 'The label must consist of {0, 1}')
assert(min(y(:))>=0, 'The label must consist of {0, 1}')
% Initialization
nrow = size(y, 1);
solution = nan(nrow, 1);
% Calculate the desired number of examples of each label for each fold
desired = repmat(sum(y)/n, n, 1);  % fold vs. label matrix
while any(isnan(solution))
    % Get the count of unnasigned positive samples for each fold
    available = sum(y(isnan(solution), :), 1);
    if max(available)==0
        solution = zerolabels(solution, n);
        return
    end
    
    % Find the label with the fewest (but at least one) remaining examples,
    % breaking ties randomly
    label = find(available == min(available(available>0)));
    label = label(randi(length(label)));
    
    for row = find(y(:, label) == 1 & isnan(solution))'
        % Find the subset(s) with the largest number of desired examples for this
        % label, breaking ties by considering the largest number of desired examples,
        % breaking further ties randomly
        fold = find(desired(:, label) == max(desired(:, label)));
        fold2 = find(sum(desired, 2) == max(sum(desired(fold, :), 2)));
        fold3 = fold2(randi(length(fold2)));
        % Update the counter
        desired(fold3, label) = desired(fold3, label) - 1;
        % Assign the sample into the fold
        solution(row) = fold3;
    end
end
% Subroutine to deal with samples that do not belong to any label.
% We just spread these samples equally
function solution = zerolabels(solution, folds)
    i = find(isnan(solution));
    solution(i) = mod1(1:length(i), folds);
end
end