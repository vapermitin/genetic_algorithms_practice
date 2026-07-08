import tkinter as tk
from tkinter import font, filedialog
from genetic_algorithm import *
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class GeneticAlgorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm")
        self.root.geometry("400x400")
        self.root.minsize(200, 200)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)

        self.n = 3
        self.vars = []
        self.entries = []

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        for i in range(6):
            btn_frame.columnconfigure(i, weight=1)
        tk.Button(btn_frame, text="Add Vertex", command=self.add_vertex).grid(row=0, column=0, padx=5, sticky="ew")
        tk.Button(btn_frame, text="Remove Vertex", command=self.remove_vertex).grid(row=0, column=1, padx=5, sticky="ew")
        tk.Button(btn_frame, text="Show Graph", command=self.show_graph).grid(row=0, column=2, padx=5, sticky="ew")
        tk.Button(btn_frame, text="Save Matrix", command=self.save_matrix).grid(row=0, column=3, padx=5, sticky="ew")
        tk.Button(btn_frame, text="Load Matrix", command=self.load_matrix).grid(row=0, column=4, padx=5, sticky="ew")
        tk.Button(btn_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).grid(row=0, column=5, padx=5, sticky="ew")

        self.frame = tk.Frame(root, bg='lightgray')
        self.frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.frame.bind("<Configure>", self.on_frame_configure)
        self.build_grid()

    def get_iterations_count(self):
        return 100

    def build_grid(self):
        for row in self.entries:
            for e in row:
                e.destroy()
        self.entries.clear()
        self.vars.clear()

        for i in range(self.n + 1):
            self.frame.grid_rowconfigure(i, weight=0, uniform='', minsize=0)
            self.frame.grid_columnconfigure(i, weight=0, uniform='', minsize=0)

        for i in range(self.n):
            self.frame.grid_rowconfigure(i, weight=1, uniform='cell')
            self.frame.grid_columnconfigure(i, weight=1, uniform='cell')

        for i in range(self.n):
            row_vars = []
            row_entries = []
            for j in range(self.n):
                var = tk.StringVar(value="0")
                var.trace_add("write", lambda *a, r=i, c=j: self.on_change(r, c))
                row_vars.append(var)
                e = tk.Entry(self.frame, textvariable=var, justify="center", width=1)
                e.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                row_entries.append(e)
            self.vars.append(row_vars)
            self.entries.append(row_entries)

        self.adjust_sizes()

    def on_frame_configure(self, event):
        self.adjust_sizes()

    def adjust_sizes(self):
        if self.n == 0:
            return

        frame_w = self.frame.winfo_width()
        frame_h = self.frame.winfo_height()

        pad = 2 * 1
        col_w = max(10, (frame_w - pad * self.n) // self.n)
        row_h = max(10, (frame_h - pad * self.n) // self.n)

        font_size = max(6, min(24, int(row_h * 0.4)))
        f = font.Font(family="TkDefaultFont", size=font_size)

        char_w = f.measure("0")
        entry_width = max(1, int(col_w / char_w))

        for row in self.entries:
            for e in row:
                e.config(font=f, width=entry_width)

    def on_change(self, i, j):
        val = self.vars[i][j].get()
        try:
            float(val)
        except ValueError:
            if val != "":
                val = "0"
                self.vars[i][j].set(val)
        if self.vars[j][i].get() != val:
            self.vars[j][i].set(val)
    
    def run_genetic_algorithm(self):
        ga = GeneticAlgorithm(
            self.get_adjacency_matrix(),
            population_size=100,
            hard_constraint_penalty=10,
            crossover_probability=0.8,
            mutation_probability=0.01,
            tournament_size=10,
            elitement_size=2
        )
        best_individual, fitnesses = ga.run(self.get_iterations_count())
        print(sum(best_individual))
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, self.get_iterations_count() + 1)), fitnesses)
        plt.show()

    def add_vertex(self):
        self.n += 1
        self.build_grid()

    def remove_vertex(self):
        if self.n > 0:
            self.n -= 1
            self.build_grid()
    
    def get_adjacency_matrix(self):
        return [[float(self.vars[i][j].get().strip()) if self.vars[i][j].get() else 0.0 for i in range(self.n)] for j in range(self.n)]

    def save_matrix(self):
        file_object = filedialog.asksaveasfile(
            mode='wb',
            defaultextension=".graph",
            filetypes=[("Graph files", "*.graph")]
        )

        if file_object is not None:
            adjacency_matrix = self.get_adjacency_matrix()
            pickle.dump(adjacency_matrix, file_object)
            file_object.close()
    
    def load_matrix(self):
        file_object = filedialog.askopenfile(
            mode='rb',
            filetypes=[("Graph files", "*.graph")]
        )

        if file_object is not None:
            adjacency_matrix = pickle.load(file_object)
            self.n = len(adjacency_matrix)
            self.build_grid()
            for i in range(self.n):
                for j in range(self.n):
                    self.vars[i][j].set(f"{adjacency_matrix[i][j]:g}")
            file_object.close()

    def show_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                val = self.vars[i][j].get().strip()
                if val and val != "0":
                    try:
                        w = float(val)
                        G.add_edge(i, j, weight=w)
                    except ValueError:
                        G.add_edge(i, j)
        plt.figure(figsize=(5,5))
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title("Graph")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmApp(root)
    root.mainloop()
