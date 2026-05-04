import osmnx as ox
print("Downloading San Francisco Map for instant loading...")
G = ox.graph_from_address("San Francisco, California", dist=15000, network_type='drive')
ox.save_graphml(G, filepath="sf_map.graphml")
print("✅ Done! 'sf_map.graphml' saved. Your app will now be instant.")
