import matplotlib.pyplot as plt

class StreetNetworkVisualization():
    def __init__(self, street_network):
        self.street_network = street_network
        self.fig, self.ax = plt.subplots(figsize=(20,20))
        self.ax.set_aspect("equal")
        
    def set_bounds(self, xmin, xmax, ymin, ymax, margin):
        self.ax.set_xlim(xmin - margin, xmax + margin)
        self.ax.set_ylim(ymin - margin, ymax + margin)

    def plot_street_network(self):
        for line in self.street_network.edges_df.line.map(lambda x: x.bounds):
            self.ax.plot([line[0], line[2]], [line[1], line[3]], color="black", alpha=0.1)