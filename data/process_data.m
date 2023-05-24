rng('shuffle')
datasets = ["DrugBank_DDI" "CTD_DDA" "NDFRT_DDA" "STRING_PPI"];

for dataset = datasets
path = strcat(dataset, '/', dataset, '.edgelist');
edgelist = readtable(path, 'FileType', 'text', 'ReadVariableNames', false);

nodes = unique([edgelist.Var1, edgelist.Var2]);
n = length(nodes);
node_map = containers.Map(nodes, (1:n));
edgelist = [edgelist.Var1, edgelist.Var2];
edgelist = arrayfun(@(x) node_map(x), edgelist);

adj = sparse(edgelist(:, 1), edgelist(:, 2), 1, n, n);
adj = adj + adj';
adj = adj - diag(diag(adj));
adj(adj~=0) = 1;

% components = conncomp(graph(adj));
% c = mode(components);
% idx = find(components ~= c);
% adj(idx, :) = [];
% adj(:, idx) = [];
% n = length(adj);

adj_triu = triu(adj);
[r, c] = find(adj_triu);
edges_all = [r, c] - 1;
num_test = fix(length(edges_all) * 0.15);
num_val = fix(length(edges_all) / 20);
all_edge_idx = randperm(length(edges_all));
val_edge_idx = all_edge_idx(1 : num_val);
test_edge_idx = all_edge_idx(num_val : (num_val + num_test));
test_edges = edges_all(test_edge_idx, :);
val_edges = edges_all(val_edge_idx, :);
train_edges = setdiff(edges_all, [test_edges; val_edges], 'rows');
adj_train = sparse(train_edges(:, 1) + 1, train_edges(:, 2) + 1, 1, n, n);
adj_train = adj_train + adj_train';

test_edges_false = [0, 0];
for i = 1 : num_test
    node1 = randi([0, n - 1]);
    node2 = randi([0, n - 1]);
    if ~ismember([node1, node2], test_edges_false, "rows") && ~ismember([node1, node2], train_edges, "rows")
        test_edges_false = [test_edges_false; node1, node2];
    end
end

val_edges_false = [0, 0];
for i = 1 : num_val
    node1 = randi([0, n - 1]);
    node2 = randi([0, n - 1]);
    if ~ismember([node1, node2], val_edges_false, "rows") && ~ismember([node1, node2], train_edges, "rows")
        val_edges_false = [val_edges_false; node1, node2];
    end
end

test_edges_false = test_edges_false(2 : length(test_edges_false), :);
val_edges_false = val_edges_false(2 : length(val_edges_false), :);

save(dataset, 'adj', 'adj_train', 'train_edges', 'val_edges', 'val_edges_false', 'test_edges', 'test_edges_false')
end
