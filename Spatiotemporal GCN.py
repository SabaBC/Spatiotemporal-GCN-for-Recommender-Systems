import os
import json
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from surprise import Dataset
from datetime import datetime
from sklearn.cluster import OPTICS
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import KFold
from networkx.algorithms import bipartite
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load and prepare data
def load_data(file_path):
    column_names = ['userId', 'itemId', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=column_names, header=None)
    return df

# Build bipartite graph with timestamps as edge extra attribute
def build_bipartite_graph(train_data):
    G = nx.Graph()
    
    users = {f'user_{u}' for u in train_data['userId']}
    items = {f'item_{i}' for i in train_data['itemId']}
    
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(items, bipartite=1)
    
    for _, row in train_data.iterrows():
        user_node = f'user_{row["userId"]}'
        item_node = f'item_{row["itemId"]}'
        G.add_edge(user_node, item_node, weight=row['rating'], timestamp=row['timestamp'])
        
    return G

# Function to add one-hot encoding to node labels
def add_one_hot_encoding_to_labels(subgraph):
    labels = [subgraph.nodes[node]['label'] for node in subgraph.nodes]
    encoder = OneHotEncoder(sparse_output=False, categories=[range(4)])
    one_hot_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1))
    
    for idx, node in enumerate(subgraph.nodes):
        subgraph.nodes[node]['one_hot_label'] = one_hot_encoded[idx]
        
    return subgraph

# Temporal based subgraph clustering: directly connected neighbors (1-hop neighbors), OPTICS clustering (maximum number of clusters), assign labels and one hot encoding
def temporal_subgraph_clustering(graph, target_user, target_item, min_samples, max_num_clusters):
    user_node = f'user_{target_user}'
    item_node = f'item_{target_item}'
    user_neighbors = set(graph.neighbors(user_node))  # Items directly connected to the target user
    item_neighbors = set(graph.neighbors(item_node))  # Users directly connected to the target item
    
    neighbors = user_neighbors.union(item_neighbors).union({user_node, item_node})
    subgraph = graph.subgraph(neighbors)
    
    # Filter edges to keep only those that involve the target user or target item (this is a way to handle 1-hop neighbors!)
    edges = [(u, v, d) for u, v, d in subgraph.edges(data=True) if user_node in [u, v] or item_node in [u, v]]
    # Sort the filtered edges by timestamp
    edges = sorted(edges, key=lambda edge: edge[2]['timestamp'])
    
    timestamps = np.array([edge[2]['timestamp'] for edge in edges]).reshape(-1, 1)
    timestamps = np.unique(timestamps).astype(np.float64).reshape(-1, 1)  # Reshape to 2D
    
    # Ensure min_samples is not greater than the number of samples (ensure formation of clusters even with fewer neighbors)
    actual_min_samples = min(min_samples, len(timestamps))
    
    if len(timestamps) > 1:
        try:
            # Add a small epsilon to avoid division by zero issues (for distance metrics calculation purposes)
            timestamps += 1e-5  # Adding epsilon
            clustering = OPTICS(min_samples=actual_min_samples).fit(timestamps)
            clusters = {}
            
            for idx, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                    
                clusters[label].append(edges[idx])
                
            # Create a list of valid (non-noise) clusters
            valid_clusters = [clusters[label] for label in clusters if label != -1]
            
        except RuntimeWarning as e:
            print(f"Warning encountered during OPTICS clustering: {e}")
            valid_clusters = [edges]
            
    else:
        valid_clusters = [edges] # If there's only one timestamp, it forms a single cluster

   # Handle cases where OPTICS forms more than maximum number of clusters (merge smaller clusters to form max_num_clusters)
    if len(valid_clusters) > max_num_clusters:
        cluster_timestamps = []
        
        for cluster in valid_clusters:
            avg_timestamp = np.mean([edge[2]['timestamp'] for edge in cluster])
            cluster_timestamps.append((avg_timestamp, cluster))
            
        cluster_timestamps = sorted(cluster_timestamps, key=lambda x: x[0])
        
        # Merge the smallest clusters until we have exactly 'max_num_clusters' clusters
        while len(cluster_timestamps) > max_num_clusters:
            # Merge the two smallest clusters
            cluster1 = cluster_timestamps.pop(0)[1]  # First cluster
            cluster2 = cluster_timestamps.pop(0)[1]  # Second cluster
            merged_cluster = cluster1 + cluster2  # Merge them
            
            # Compute the new average timestamp for the merged cluster
            new_avg_timestamp = np.mean([edge[2]['timestamp'] for edge in merged_cluster])
            # Insert the merged cluster back into the list (keeping it sorted by average timestamp)
            cluster_timestamps.append((new_avg_timestamp, merged_cluster))
            cluster_timestamps = sorted(cluster_timestamps, key=lambda x: x[0])
            
        valid_clusters = [cluster for _, cluster in cluster_timestamps]

    # Label the nodes and apply one-hot encoding
    labeled_subgraphs = []
    for cluster in valid_clusters:
        subgraph_with_labels = nx.Graph()
        subgraph_with_labels.add_node(user_node, label=0) #label target user as 0
        subgraph_with_labels.add_node(item_node, label=1) #label target item as 1
        for edge in cluster:
            node_u, node_v, edge_data = edge

            for node in [node_u, node_v]:
                if node not in subgraph_with_labels:
                    node_label = 2 if node.startswith('user') else 3  # Label neighbor item nodes with 2 and neighbor user nodes with 3
                    subgraph_with_labels.add_node(node, label=node_label)

            subgraph_with_labels.add_edge(node_u, node_v, **edge_data)

        subgraph_with_labels = add_one_hot_encoding_to_labels(subgraph_with_labels)
        labeled_subgraphs.append(subgraph_with_labels)

    return labeled_subgraphs

class TemporalMatrixLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional, batch_size):
        super(TemporalMatrixLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if bidirectional:
            self.reduce_dim = nn.Linear(2 * hidden_dim, hidden_dim) # Reduce from 64 to 32 if bidirectional
    def forward(self, temporal_embeddings):
        num_directions = 2 if self.bidirectional else 1
        batch_size = temporal_embeddings.size(0)
        padding_size = batch_size - (temporal_embeddings.size(1) % batch_size) if temporal_embeddings.size(1) % batch_size != 0 else 0
        
        if padding_size > 0:
            padding = torch.zeros((temporal_embeddings.size(0), padding_size, temporal_embeddings.size(2)))
            temporal_embeddings = torch.cat([temporal_embeddings, padding], dim=1)
        
        h_0 = torch.zeros(num_directions, batch_size, self.hidden_dim)
        c_0 = torch.zeros(num_directions, batch_size, self.hidden_dim)
        output, _ = self.lstm(temporal_embeddings, (h_0, c_0))
        
        if padding_size > 0:
            output = output[:, :-padding_size]
        if self.bidirectional:
            output = self.reduce_dim(output)
        
        temporal_matrix = torch.mean(output, dim=1)
        return temporal_matrix

# RGCN model with temporal adjacency matrix integration
class TemporalRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_layers, bidirectional, batch_size):
        super(TemporalRGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer: from input to hidden
        self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations))
        
        # Add additional hidden layers
        for _ in range(1, num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            
        # Last layer: from hidden to output
        self.convs.append(RGCNConv(hidden_dim, output_dim, num_relations))
        self.temporal_matrix_lstm = TemporalMatrixLSTM(hidden_dim, hidden_dim, bidirectional, batch_size)

    def forward(self, spatial_embeddings, edge_index, edge_type, temporal_embeddings):
        # Calculate temporal matrix from LSTM
        temporal_matrix = self.temporal_matrix_lstm(temporal_embeddings)
        # Step: Compute the dot product of spatial_embeddings and temporal_embeddings
        temporal_coefficients = torch.matmul(spatial_embeddings, temporal_matrix.T)
        # Step: Apply softmax to get probability distribution over time intervals
        temporal_coefficients = F.softmax(temporal_coefficients, dim=-1)
        layer_outputs = [spatial_embeddings]  # Start with pre-computed spatial embeddings
        
        for i in range(1, self.num_layers - 1):  # Start from layer 1 (skip the first layer)
            # Apply the RGCN layers starting from the second layer
            x = self.convs[i](spatial_embeddings, edge_index, edge_type)
            # Multiply the output of RGCN convolution by the temporal matrix
            x = torch.tanh(x * temporal_coefficients)  # Spatial adjacency matrix (via edge_index) and temporal matrix combined
            layer_outputs.append(x)
            
        # Final layer without activation
        x = self.convs[-1](x, edge_index, edge_type)
        layer_outputs.append(x)
        
        # Concatenate the feature vectors from all layers (4*32)
        final_embeddings = torch.cat(layer_outputs, dim=-1)
        
        return final_embeddings

# Subgraph embedding construction (spatiotemporal relational graph convlutional network and LSTM)
def subgraph_embedding_construction(labeled_subgraphs, rgcn_model, target_user, target_item, num_relations):
    user_item_embeddings = []
    
    for subgraph in labeled_subgraphs:
        node_to_idx = {node: idx for idx, node in enumerate(subgraph.nodes)}
        node_embeddings = np.array([subgraph.nodes[node]['one_hot_label'] for node in subgraph.nodes])
        node_embeddings = torch.FloatTensor(node_embeddings)
        
        edge_index = []
        edge_type = []
        
        for edge in subgraph.edges(data=True):
            u, v, d = edge
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_type.append(int(d['weight']) - 1)
            
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_type = torch.LongTensor(edge_type)

        # First layer of RGCN (spatial info) convs[0] focuses on propagating info to neighbors 
        spatial_embeddings = rgcn_model.convs[0](node_embeddings, edge_index, edge_type)

        user_neighbors = list(subgraph.neighbors(f'user_{target_user}'))
        item_neighbors = list(subgraph.neighbors(f'item_{target_item}'))
        
        # Fetch the temporal ordered neighbors of target user and item
        if len(user_neighbors) == 0:
            temporal_order_u = torch.zeros((1, spatial_embeddings.shape[1]))  # Zero tensor if no neighbors found for user
        else:
            temporal_order_u = torch.stack([spatial_embeddings[node_to_idx[neighbor]] for neighbor in user_neighbors])

        if len(item_neighbors) == 0:
            temporal_order_v = torch.zeros((1, spatial_embeddings.shape[1]))  # Zero tensor if no neighbors found for item
        else:
            temporal_order_v = torch.stack([spatial_embeddings[node_to_idx[neighbor]] for neighbor in item_neighbors])

        # Concat the temporal ordered neighbors for matrix multiplication
        temporal_embeddings = torch.cat([temporal_order_u.unsqueeze(0), temporal_order_v.unsqueeze(0)], dim=1)
        # Final RGCN with temporal matrix
        final_embeddings = rgcn_model(spatial_embeddings, edge_index, edge_type, temporal_embeddings)

        target_user_embedding = final_embeddings[node_to_idx[f'user_{target_user}']]
        target_item_embedding = final_embeddings[node_to_idx[f'item_{target_item}']]
        concatenated_embeddings = torch.cat([target_user_embedding, target_item_embedding], dim=-1)
        user_item_embeddings.append(concatenated_embeddings)

    return user_item_embeddings

class SelfAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionNetwork, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale_factor = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, subgraph_embeddings):
        # Compute Query, Key, and Value matrices
        Q = self.query(subgraph_embeddings)
        K = self.key(subgraph_embeddings)
        V = self.value(subgraph_embeddings)

        # Compute attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of the values (V)
        weighted_sum = torch.matmul(attention_weights, V)
        
        return weighted_sum

# MLP for Final Rating Prediction
class RatingPredictionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RatingPredictionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Activation function
            nn.Dropout(p=0.7),  # A regularization technique to prevent overfitting with 70% probability
            nn.Linear(hidden_dim, 1)  # Final output layer for rating prediction
        )

    def forward(self, aggregated_embedding):
        rating = self.mlp(aggregated_embedding)
        rating = torch.sigmoid(rating) * 4 + 1  # Sigmoid output scaled to [1, 5]
        
        return rating

def subgraphs_aggregation(user_item_embeddings_all, self_attention_net):
    final_personalized_embeddings = []

    for user_item_pair, user_item_embeddings in user_item_embeddings_all.items():
        # Convert the list of embeddings into a tensor
        user_item_embeddings_tensor = torch.stack(user_item_embeddings)
        
        # Apply self-attention to subgraph embeddings
        weighted_embeddings = self_attention_net(user_item_embeddings_tensor)

        # Aggregate into a single embedding (mean pooling)
        aggregated_embedding = torch.mean(weighted_embeddings, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Append final personalized embeddings
        final_personalized_embeddings.append(aggregated_embedding)
    # Stack all embeddings
    final_personalized_embeddings = torch.stack(final_personalized_embeddings)

    return final_personalized_embeddings

class UnifiedModel(nn.Module):
    def __init__(self, rgcn_model, self_attention_net, mlp_model, min_samples, num_clusters, num_relations):
        super(UnifiedModel, self).__init__()
        self.rgcn_model = rgcn_model
        self.self_attention_net = self_attention_net
        self.mlp_model = mlp_model
        self.min_samples = min_samples
        self.num_clusters = num_clusters
        self.num_relations = num_relations
    
    def forward(self, bipartite_graph, user_item_pairs, save_embeddings=False):
        user_item_embeddings_all = {}
        
        for target_user, target_item in user_item_pairs:
            
            labeled_subgraphs = temporal_subgraph_clustering(
                bipartite_graph, target_user, target_item, min_samples=self.min_samples, max_num_clusters=self.num_clusters
            )
            
            user_item_embeddings = subgraph_embedding_construction(
                labeled_subgraphs, self.rgcn_model, target_user, target_item, self.num_relations
            )
            
            user_item_embeddings_all[(target_user, target_item)] = user_item_embeddings
        
        final_personalized_embeddings = subgraphs_aggregation(
            user_item_embeddings_all, self.self_attention_net
        )

        if save_embeddings:
            # Save final embeddings for later use
            self.saved_embeddings = final_personalized_embeddings.squeeze(1).clone().detach()
        
        # Predict the final rating
        predicted_rating = self.mlp_model(final_personalized_embeddings).mean(dim=0)
        
        return predicted_rating
    
def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return len(recommended_at_k & relevant_items) / k

def recall_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return len(recommended_at_k & relevant_items) / len(relevant_items) if relevant_items else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    dcg = 0
    idcg = 0
    relevant_items_set = set(relevant_items)

    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items_set:
            dcg += 1 / np.log2(i + 2)  # +2 because i starts from 0
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0

def calculate_coverage(recommended_items, total_items):
    unique_recommended = set(recommended_items)
    return len(unique_recommended) / total_items

def calculate_hit_rate(recommended_items, relevant_items, k):
    # Get the top-k recommended items
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    # Check if there is any overlap between recommended and relevant items
    hit = 1 if recommended_at_k & relevant_set else 0
    return hit

# Train Function
def train_model_batched(bipartite_graph, train_data, unified_model, optimizer, batch_size, l2_lambda):
    unified_model.train()  # Set the model to training mode
    total_loss = 0.0
    total_ratings_count = 0
    start_time = time.time()  # Track overall start time
    last_report_time = start_time  # Track the last time progress was reported
    rmse_running_sum = 0.0  # To accumulate RMSE over batches for reporting
    batch_count = 0
    user_item_index_embeddings_train = []  # List to store user-item indices and embeddings
    
    for batch_start in range(0, len(train_data), batch_size):
        batch_data = train_data.iloc[batch_start:batch_start + batch_size]
        batch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients for the batch

        # Store predicted ratings and true ratings for each user-item pair in the batch
        predicted_ratings = []
        true_ratings = []

        for _, row in batch_data.iterrows():
            target_user = row['userId']
            target_item = row['itemId']
            true_rating = torch.tensor([row['rating']]).float()

            # Predicted rating for the current user-item pair
            predicted_rating = unified_model(bipartite_graph, [(target_user, target_item)], save_embeddings=True)     

            # Add userId, itemId, and embeddings to the list
            user_item_index_embeddings_train.append({
                "userId": int(target_user),  # Convert numpy.int64 to standard Python int
                "itemId": int(target_item),  # Convert numpy.int64 to standard Python int
                "embedding": unified_model.saved_embeddings.cpu().numpy().tolist()  # Convert tensor to list without squeezing
            })

            # Average the predicted ratings if there are multiple subgraphs
            predicted_rating = predicted_rating.mean(dim=0)  # Reduce to a single rating

            predicted_ratings.append(predicted_rating)
            true_ratings.append(true_rating)
            
        # Stack the predicted ratings and true ratings for the batch
        predicted_ratings = torch.stack(predicted_ratings).squeeze()
        true_ratings = torch.stack(true_ratings).squeeze()

        # Compute the MSE loss for the batch
        mse_loss = F.mse_loss(predicted_ratings, true_ratings, reduction='sum')
        # Compute L2 regularization for all submodules (Loss function for model training)
        l2_regularization = l2_lambda * torch.sum(torch.cat([param.view(-1).pow(2) for param in unified_model.parameters()]))
        # Final loss for the batch
        loss = mse_loss + l2_regularization

        # Backpropagate the batch loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate total loss for reporting
        total_ratings_count += len(batch_data)  # Count the number of processed ratings
        rmse_running_sum += torch.sqrt(F.mse_loss(predicted_ratings, true_ratings)).item()
        batch_count += 1

        # Calculate elapsed time and print progress after every 20 minutes
        current_time = time.time()
        elapsed_time = current_time - last_report_time

        if elapsed_time > 20 * 60:  # If more than 20 minutes have passed
            # Calculate current progress and average RMSE so far
            progress_percentage = (batch_start / len(train_data)) * 100
            avg_rmse = rmse_running_sum / batch_count  # Average RMSE across batches
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"{progress_percentage:.2f}% completed at {current_timestamp}.")
            print(f"Running RMSE: {avg_rmse:.4f}")
            
            # Reset last report time for next 20-minute check
            last_report_time = current_time

    with open('//train_embeddings.json', 'w') as f:
        json.dump(user_item_index_embeddings_train, f)
    print(f"Saved user-item embeddings to train_embeddings.json")    
    
    # Compute the average loss over the entire training set
    average_loss = total_loss / total_ratings_count
    print(f"Average training loss (with regularization): {average_loss}")

# Test Function
def test_model(bipartite_graph, test_data, unified_model, metric_k):
    unified_model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_count = 0
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_count = 0
    all_recommended_items = set()
    all_items = set(test_data['itemId'].unique())

    start_time = time.time()  # Track overall start time
    last_report_time = start_time  # Track the last time progress was reported
    user_count = len(test_data['userId'].unique())
    processed_users = 0

    # New: Store embeddings during testing
    user_item_index_embeddings_test = []

    with torch.no_grad():
        for user in test_data['userId'].unique():
            # Increment processed users
            processed_users += 1
            
            # Get all interactions for the user
            user_data = test_data[test_data['userId'] == user]

            # Predict scores for all items
            all_item_scores = []
            for item in all_items:
                predicted_rating = unified_model(bipartite_graph, [(user, item)], save_embeddings=True)
                predicted_rating = predicted_rating.mean().item()  # Ensure single value
                all_item_scores.append((item, predicted_rating))
                
                # Save test embeddings
                user_item_index_embeddings_test.append({
                    "userId": int(user),
                    "itemId": int(item),
                    "embedding": unified_model.saved_embeddings.cpu().numpy().tolist()
                })
                
            # Rank items by predicted scores
            all_item_scores = sorted(all_item_scores, key=lambda x: x[1], reverse=True)
            recommended_items = [x[0] for x in all_item_scores]

            # Relevant items for the user (test set)
            relevant_items = user_data['itemId'].tolist()

            # Calculate ranking-based metrics
            precision_scores.append(precision_at_k(recommended_items, relevant_items, metric_k))
            recall_scores.append(recall_at_k(recommended_items, relevant_items, metric_k))
            ndcg_scores.append(ndcg_at_k(recommended_items, relevant_items, metric_k))
            all_recommended_items.update(recommended_items[:metric_k])
            hit_count += calculate_hit_rate(recommended_items, relevant_items, metric_k)

            # Calculate RMSE and MAE for this user
            for _, row in user_data.iterrows():
                true_rating = torch.tensor([row['rating']]).float()
                predicted_rating = unified_model(bipartite_graph, [(row['userId'], row['itemId'])])
                if predicted_rating.dim() > 1:
                    predicted_rating = predicted_rating.mean(dim=0)
                predicted_rating = predicted_rating.view(-1)  # Ensure 1D tensor
                true_rating = true_rating.view(-1)  # Ensure 1D tensor

                squared_error = F.mse_loss(predicted_rating, true_rating).item()
                total_squared_error += squared_error

                absolute_error = F.l1_loss(predicted_rating, true_rating, reduction='sum').item()
                total_absolute_error += absolute_error
                total_count += 1

            # Periodically print progress every 20 minutes
            current_time = time.time()
            elapsed_time = current_time - last_report_time

            if elapsed_time > 20 * 60:  # If more than 20 minutes have passed
                progress_percentage = (processed_users / user_count) * 100                
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{progress_percentage:.2f}% completed at {current_timestamp}.")
                last_report_time = current_time  # Reset the report timer
    
    # Save the test embeddings to a JSON file
    with open('//test_embeddings.json', 'w') as f:
        json.dump(user_item_index_embeddings_test, f)
    
    print(f"Saved test user-item embeddings to test_embeddings.json")

    # Compute mean metrics
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_ndcg = np.mean(ndcg_scores)

    # Calculate coverage
    coverage = calculate_coverage(all_recommended_items, len(all_items))

    # Calculate HR@K
    hit_rate = hit_count / len(test_data['userId'].unique())

    rmse = torch.sqrt(torch.tensor(total_squared_error / total_count)).item()
    mae = total_absolute_error / total_count

    return rmse, mae, mean_precision, mean_recall, mean_ndcg, coverage, hit_rate

# Main Model Function (for 80/20 split)
def Model_80_20(skip_test=True):
    # Parameters
    L2_LAMBDA = 0.005  #L2 Regularization
    METRIC_K = 10 # K for evaluation metrics
    INPUT_DIM = 4  # One-hot Encoding dimension
    LEARNING_RATE = 1e-3  # Optimizer Learning Rate
    HIDDEN_DIM = 32  # Optimal hidden Dimension of RGCN
    OUTPUT_DIM = 32  # Optimal output dimension of RGCN
    NUM_LAYERS = 4  # Optimal number of layers for RGCN
    BATCH_SIZE = 32  # Batch size for parallel processing
    NUM_RELATIONS = 5  # 5 ratings --> 5 types of relations for RGCN
    MIN_SAMPLES = 10  # Optimal minPts for OPTICS clustering algorithm
    BIDIRECTIONAL_LSTM = True  # Defines whether LSTM is bidirectional or not
    NUMBER_OF_CLUSTERS = 5  # Optimal number of temporal clusters for each user item pair
    FILE_PATH = '//Dataset//ml-100k//u.data'  # Dataset file path
    EMBEDDING_DIM = 256  # Embedding dimension of RGCN (256=2*128 --> user and item concatenation) (128=4*32 --> 4 hidden layers, each 32 dimension)

    # Load the dataset
    data = load_data(FILE_PATH)
    ###data = data.sample(n=10)
    
    # Unique users in the dataset
    unique_users = data['userId'].unique()
    
    # Split users into train and test groups
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    
    # Select rows where userId is in train_users for training data, and test_users for testing data
    train_data = data[data['userId'].isin(train_users)]
    test_data = data[data['userId'].isin(test_users)]

    # Build bipartite graphs for train and test data
    train_bipartite_graph = build_bipartite_graph(train_data)
    test_bipartite_graph = build_bipartite_graph(test_data)
    
    # Make Instances of All Models
    rgcn_model = TemporalRGCN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_RELATIONS, NUM_LAYERS, BIDIRECTIONAL_LSTM, BATCH_SIZE)
    mlp_model = RatingPredictionMLP(HIDDEN_DIM, HIDDEN_DIM)
    self_attention_net = SelfAttentionNetwork(EMBEDDING_DIM, HIDDEN_DIM)
    unified_model = UnifiedModel(
        rgcn_model, self_attention_net, mlp_model, 
        min_samples=MIN_SAMPLES, num_clusters=NUMBER_OF_CLUSTERS, 
        num_relations=NUM_RELATIONS
    )
    
    # Optimizer for the main unified model
    optimizer = optim.Adam(unified_model.parameters(), lr=LEARNING_RATE)

    # Train the model on the training data
    print(f"Training the model...")
    train_model_batched(train_bipartite_graph, train_data, unified_model, optimizer, BATCH_SIZE, L2_LAMBDA)
    '''
    if skip_test:  # Check the flag before proceeding to the test phase
        print("Skipping the test phase as per the configuration.")
        return  # Exit the function early if the flag is set
    '''
    # Test the model on the test data
    print(f"Evaluating the model on test data...")
    rmse, mae, precision, recall, ndcg, coverage, hit_rate = test_model(test_bipartite_graph, test_data, unified_model, METRIC_K)
    
    # Print metrics
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Precision@{METRIC_K}: {precision:.4f}")
    print(f"Test Recall@{METRIC_K}: {recall:.4f}")
    print(f"Test NDCG@{METRIC_K}: {ndcg:.4f}")
    print(f"Test Coverage: {coverage:.4f}")
    print(f"Test HR@{METRIC_K}: {hit_rate:.4f}")
    
def main():
    start_time = time.time()
    start_time_readable = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Execution started at: {start_time_readable}")

    Model_80_20()
    
    end_time = time.time()
    end_time_readable = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    execution_time = end_time - start_time
    print(f"Execution ended at: {end_time_readable}")
    print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()