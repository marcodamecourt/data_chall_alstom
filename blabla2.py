import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Définir le DataFrame avec plusieurs sous-réseaux interconnectés
data = {
    'Station_A': [
        None, {'lines': ['Ligne 1'], 'distance': 1.0, 'passengers': 200}, None, {'lines': ['Ligne 2'], 'distance': 2.5, 'passengers': 150}, None, None, None, None, None, None],
    'Station_B': [
        {'lines': ['Ligne 1'], 'distance': 1.0, 'passengers': 200}, None, {'lines': ['Ligne 1'], 'distance': 1.5, 'passengers': 300}, None, None, None, None, None, None, None],
    'Station_C': [
        None, {'lines': ['Ligne 1'], 'distance': 1.5, 'passengers': 300}, None, {'lines': ['Ligne 3'], 'distance': 1.0, 'passengers': 250}, None, None, None, None, None, None],
    'Station_D': [
        {'lines': ['Ligne 2'], 'distance': 2.5, 'passengers': 150}, None, {'lines': ['Ligne 3'], 'distance': 1.0, 'passengers': 250}, None, {'lines': ['Ligne 4'], 'distance': 1.5, 'passengers': 180}, None, None, None, None, None],
    'Station_E': [
        None, None, None, {'lines': ['Ligne 4'], 'distance': 1.5, 'passengers': 180}, None, {'lines': ['Ligne 5'], 'distance': 2.0, 'passengers': 200}, None, None, None, None],
    'Station_F': [
        None, None, None, None, {'lines': ['Ligne 5'], 'distance': 2.0, 'passengers': 200}, None, {'lines': ['Ligne 6'], 'distance': 1.8, 'passengers': 150}, None, None, None],
    'Station_G': [
        None, None, None, None, None, {'lines': ['Ligne 6'], 'distance': 1.8, 'passengers': 150}, None, {'lines': ['Ligne 7'], 'distance': 1.2, 'passengers': 100}, None, None],
    'Station_H': [
        None, None, None, None, None, None, {'lines': ['Ligne 7'], 'distance': 1.2, 'passengers': 100}, None, {'lines': ['Ligne 8'], 'distance': 2.5, 'passengers': 250}, None],
    'Station_I': [
        None, None, None, None, None, None, None, {'lines': ['Ligne 8'], 'distance': 2.5, 'passengers': 250}, None, {'lines': ['Ligne 9'], 'distance': 1.0, 'passengers': 180}],
    'Station_J': [
        None, None, None, None, None, None, None, None, {'lines': ['Ligne 9'], 'distance': 1.0, 'passengers': 180}, None]
}

# Liste des stations, en s'assurant qu'il y a bien 10 stations
stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E', 'Station_F', 'Station_G', 'Station_H', 'Station_I', 'Station_J']

# Vérification que toutes les stations ont une liaison avec toutes les autres (ou None si pas de connexion)
graph_df = pd.DataFrame(data, index=stations)

# Affichage du DataFrame pour voir les liaisons
print(graph_df)

# Création du graphe avec NetworkX à partir du DataFrame
G = nx.DiGraph()

# Ajouter des arêtes au graphe en utilisant les informations du DataFrame
for start_station in graph_df.index:
    for end_station in graph_df.columns:
        connection = graph_df.loc[start_station, end_station]
        if connection is not None:
            G.add_edge(start_station, end_station,
                       lines=connection['lines'],
                       distance=connection['distance'],
                       passengers=connection['passengers'])

start_station = 'Station_A'
end_station = 'Station_G'
# Trouver tous les chemins entre Station_A et Station_F
all_paths = list(nx.all_simple_paths(G, source=start_station, target=end_station))


# Afficher les chemins avec des informations détaillées
print(f"Tous les chemins de {start_station} à {end_station}:")
for path in all_paths:
    # Calculer la distance totale et d'autres métriques si nécessaire
    total_distance = 0
    total_passengers = 0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_distance += edge_data['distance']
        total_passengers += edge_data['passengers']
    print(f"Chemin: {path}, Distance totale: {total_distance} km, Passagers: {total_passengers}")

# Dessiner le graphe
pos = nx.spring_layout(G)  # Positionnement automatique des nœuds

plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, edge_color="gray", font_size=10, font_weight="bold")

# Ajouter les labels des arêtes avec les informations (distance et passagers)
edge_labels = {(u, v): f"{d['distance']} km\n{d['passengers']} pass."
               for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Réseau de Métro avec Liaisons et Passagers")
plt.show()
