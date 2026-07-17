import tkinter as tk
from tkinter import font, filedialog, messagebox
import networkx as nx
from genetic_algorithm import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import random
from math import inf, copysign

# TODO: добавить ввод вероятности появления ребра, добавить возможность указать количество вершин, убрать ребра на диагонали, добавить сохранение в json
GRID_OPTIMIZATION = True

def on_entry_click(event, placeholder_text):
    entry = event.widget
    if entry.get() == placeholder_text:
        entry.delete(0, tk.END)
        entry.config(fg='black')

def on_focusout(event, placeholder_text):
    entry = event.widget
    if entry.get() == "":
        entry.insert(0, placeholder_text)
        entry.config(fg='grey')

class GeneticAlgorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm")
        self.root.geometry("900x650")
        self.root.minsize(900, 650)

        root.columnconfigure(0, weight=1, uniform='col')
        root.columnconfigure(1, weight=1, uniform='col')
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(3, weight=0)
        root.rowconfigure(4, weight=0)

        self.n = 3
        self.vars = []
        self.entries = []

        self.graph_settings_frame = tk.Frame(root)
        self.graph_settings_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        for i in range(3):
            self.graph_settings_frame.columnconfigure(i, weight=1)
        tk.Button(self.graph_settings_frame, text="Add Vertex", command=self.add_vertex).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.graph_settings_frame, text="Remove Vertex", command=self.remove_vertex).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.graph_settings_frame, text="Show Graph", command=self.show_graph).grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.graph_settings_frame, text="Save Matrix", command=self.save_matrix).grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.graph_settings_frame, text="Load Matrix", command=self.load_matrix).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.graph_settings_frame, text="Random Graph", command=self.generate_random_graph).grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.add_textbox(self.graph_settings_frame, 2, 0, "Edge probability", "edge_probability_entry")
        self.add_textbox(self.graph_settings_frame, 2, 1, "Vertex count", "vertex_count_entry")
        tk.Button(self.graph_settings_frame, text="Set Vertex Count", command=self.set_vertex_count).grid(row=2, column=2, padx=5, pady=5, sticky="nsew")

        self.graph_frame = tk.Frame(root, bg='white')
        self.graph_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fitness_graph_frame = tk.Frame(root, bg='white')
        self.fitness_graph_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 4))
        self.ax2.clear()
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.fitness_graph_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Fitness")

        self.root.update_idletasks()
        self.fig.tight_layout()
        self.fig2.tight_layout()
        self.canvas.draw()
        self.canvas2.draw()

        self.parameters_frame = tk.Frame(root)
        self.parameters_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        for i in range(3):
            self.parameters_frame.columnconfigure(i, weight=1)

        self.add_textbox(self.parameters_frame, 0, 0, "Population size", "population_size_entry")
        self.add_textbox(self.parameters_frame, 0, 1, "Hard Constraint Penalty", "hard_constraint_penalty_entry")
        self.add_textbox(self.parameters_frame, 0, 2, "Crossover probability", "crossover_probability_entry")
        self.add_textbox(self.parameters_frame, 1, 0, "Mutation Probability", "mutation_probability_entry")
        self.add_textbox(self.parameters_frame, 1, 1, "Tournament size", "tournament_size_entry")
        self.add_textbox(self.parameters_frame, 1, 2, "Elitement size", "elitement_count_entry")
        self.add_textbox(self.parameters_frame, 2, 0, "Iterations count", "iterations_entry", columnspan=3)

        control_frame = tk.Frame(root)
        control_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        control_frame.columnconfigure(4, weight=1)

        tk.Button(control_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).grid(
            row=0, column=0, padx=5, sticky="ew")
        tk.Button(control_frame, text="Prev Iteration", command=self.prev_iteration).grid(
            row=0, column=1, padx=5, sticky="ew")
        tk.Button(control_frame, text="Next Iteration", command=self.next_iteration).grid(
            row=0, column=2, padx=5, sticky="ew")
        tk.Button(control_frame, text="Reset", command=self.reset_algorithm).grid(
                  row=0, column=3, padx=5, sticky="ew")

        self.iteration_var = tk.StringVar(value="Iteration: -/-")
        tk.Label(control_frame, textvariable=self.iteration_var).grid(
            row=0, column=4, padx=5)

        self.frame = tk.Frame(root, bg='lightgray')
        self.frame.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=5)
        self.frame.bind("<Configure>", self.on_frame_configure)
        self.build_grid()

        self.ga = None
        self.max_iterations = 0
        self.current_iteration = 0
        self.populations = []
        self.best_individuals = []
        self.best_fitnesses = []

    def add_textbox(self, parent_widget, row, column, placeholder, attribute_name, **grid_kwargs):
        entry = tk.Entry(parent_widget, fg='grey')
        entry.insert(0, placeholder)
        entry.bind('<FocusIn>', lambda e: on_entry_click(e, placeholder))
        entry.bind('<FocusOut>', lambda e: on_focusout(e, placeholder))
        default_grid = {'sticky': 'nsew', 'padx': 5, 'pady': 5}
        default_grid.update(grid_kwargs)
        entry.grid(row=row, column=column, **default_grid)
        setattr(self, attribute_name, entry)

    def destroy_grid(self):
        self.reset_algorithm()
        for row in self.entries:
            for e in row:
                e.destroy()
        self.entries.clear()
        self.vars.clear()

        for i in range(self.n + 1):
            self.frame.grid_rowconfigure(i, weight=0, uniform='', minsize=0)
            self.frame.grid_columnconfigure(i, weight=0, uniform='', minsize=0)

    def build_grid(self):
        self.destroy_grid()

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
                if self.n < 100 and GRID_OPTIMIZATION:
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
            float_value = float(val)
            if float_value < 0 or copysign(1.0, float_value) < 0:
                val = "0"
                self.vars[i][j].set(val)
        except ValueError:
            if val != "":
                val = "0"
                self.vars[i][j].set(val)
        self.reset_algorithm()
        if self.vars[j][i].get() != val:
            self.vars[j][i].set(val)

    def get_adjacency_matrix(self):
        return [[float(self.vars[i][j].get().strip()) if self.vars[i][j].get() else 0.0
                 for i in range(self.n)] for j in range(self.n)]

    def check_parameters(self):
        try:
            population_size = int(self.population_size_entry.get())
            hard_constraint_penalty = int(self.hard_constraint_penalty_entry.get())
            crossover_probability = float(self.crossover_probability_entry.get())
            mutation_probability = float(self.mutation_probability_entry.get())
            tournament_size = int(self.tournament_size_entry.get())
            elitement_count = int(self.elitement_count_entry.get())
            iterations_count = int(self.iterations_entry.get())
            if not (population_size > 0):
                return False
            if not (hard_constraint_penalty > 1):
                return False
            if not (0 <= crossover_probability <= 1):
                return False
            if not (0 <= mutation_probability <= 1):
                return False
            if not (tournament_size > 0) or not (elitement_count > 0) or not (iterations_count > 0):
                return False
        except ValueError:
            return False
        return True

    def _initialize_ga(self):
        if not self.check_parameters():
            messagebox.showerror(title="Application Error", message="Bad parameters!")
            return False

        matrix = self.get_adjacency_matrix()
        pop_size = int(self.population_size_entry.get())
        hard_pen = int(self.hard_constraint_penalty_entry.get())
        cross_prob = float(self.crossover_probability_entry.get())
        mut_prob = float(self.mutation_probability_entry.get())
        tour_size = int(self.tournament_size_entry.get())
        elite_size = int(self.elitement_count_entry.get())
        self.max_iterations = int(self.iterations_entry.get())

        self.ga = GeneticAlgorithm(matrix, pop_size, hard_pen, cross_prob, mut_prob, tour_size, elite_size)

        init_pop = self.ga.generate_random_population(pop_size)
        init_fitnesses = [self.ga.fitness_function(ind) for ind in init_pop]
        best_idx = max(range(len(init_pop)), key=lambda i: init_fitnesses[i])

        self.populations = [init_pop]
        self.best_individuals = [init_pop[best_idx].copy()]
        self.best_fitnesses = [init_fitnesses[best_idx]]
        self.current_iteration = 0

        self.update_display()
        return True

    def run_genetic_algorithm(self):
        if not self._initialize_ga():
            return

        for i in range(self.max_iterations):
            self._step_forward()

        self.current_iteration = self.max_iterations
        self.update_display()

    def _step_forward(self):
        last_pop = self.populations[-1]
        new_pop = self.ga.run_iteration(last_pop)
        new_fitnesses = [self.ga.fitness_function(ind) for ind in new_pop]
        best_idx = max(range(len(new_pop)), key=lambda i: new_fitnesses[i])
        new_best_fitness = new_fitnesses[best_idx]

        prev_best_fitness = self.best_fitnesses[-1] if self.best_fitnesses else -inf
        if new_best_fitness > prev_best_fitness:
            new_best_individual = new_pop[best_idx].copy()
        else:
            new_best_individual = self.best_individuals[-1].copy()

        self.populations.append(new_pop)
        self.best_individuals.append(new_best_individual)
        self.best_fitnesses.append(new_best_fitness)

    def set_vertex_count(self):
        try:
            if int(self.vertex_count_entry.get()) < 0:
                messagebox.showerror(title="Application Error", message="Negative vertex count!")
                return
        except ValueError:
            messagebox.showerror(title="Application Error", message="Incorrect vertex count value!")
            return
        self.destroy_grid()
        self.n = int(self.vertex_count_entry.get())
        self.build_grid()

    def reset_algorithm(self):
        self.ga = None
        self.max_iterations = 0
        self.current_iteration = 0
        self.populations = []
        self.best_individuals = []
        self.best_fitnesses = []

        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw()
        self.ax2.clear()
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Fitness")
        self.canvas2.draw()

        self.iteration_var.set("Iteration: -/-")

    def next_iteration(self):
        if self.ga is None:
            if not self._initialize_ga():
                return

        if self.current_iteration < self.max_iterations:
            if self.current_iteration == len(self.populations) - 1:
                self._step_forward()
            self.current_iteration += 1
            self.update_display()

    def prev_iteration(self):
        if self.ga is None:
            return
        if self.current_iteration > 0:
            self.current_iteration -= 1
            self.update_display()

    def update_display(self):
        self.iteration_var.set(f"Iteration: {self.current_iteration}/{self.max_iterations}")

        self.ax2.clear()
        if self.best_fitnesses:
            x = range(self.current_iteration + 1)
            y = self.best_fitnesses[:self.current_iteration + 1]
            self.ax2.plot(x, y, marker='o')
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Fitness")
        self.canvas2.draw()

        if self.best_individuals and 0 <= self.current_iteration < len(self.best_individuals):
            individual = self.best_individuals[self.current_iteration]
            self.show_graph()
        else:
            self.ax.clear()
            self.canvas.draw()

    def add_vertex(self):
        self.n += 1
        self.build_grid()

    def remove_vertex(self):
        if self.n > 0:
            self.n -= 1
            self.build_grid()

    def generate_random_graph(self):
        edge_probability = None
        try:
            edge_probability = float(self.edge_probability_entry.get())
        except ValueError:
            messagebox.showerror(title="Application Error", message="Edge probability is not float!")
            return
        if edge_probability < 0 or edge_probability > 1:
            messagebox.showerror(title="Application Error", message="Bad edge probability!")
            return
        old_reset_algorithm = self.reset_algorithm
        self.reset_algorithm = lambda: None
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.vars[i][j].set(0)
                    continue
                if random.random() < edge_probability:
                    self.vars[i][j].set(1)
                else:
                    self.vars[i][j].set(0)
        self.reset_algorithm = old_reset_algorithm
        self.reset_algorithm()

    def save_matrix(self):
        file_object = filedialog.asksaveasfile(
            mode='w',
            defaultextension=".graph",
            filetypes=[("Graph files", "*.graph")]
        )
        if file_object is not None:
            adjacency_matrix = self.get_adjacency_matrix()
            json.dump(adjacency_matrix, file_object, indent=4)
            file_object.close()

    def check_adjacency_matrix(self, matrix):
        n = len(matrix)
        if not isinstance(matrix, list):
            return False
        for i in range(self.n):
            if not (0 <= i < len(matrix)):
                return False
            if not isinstance(matrix[i], list):
                return False
            for j in range(self.n):
                if not (0 <= j < len(matrix[i])) or not (0 <= j < len(matrix)):
                    return False
                if not (0 <= i < len(matrix[j])):
                    return False
                if not isinstance(matrix[i][j], float):
                    return False
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True

    def load_matrix(self):
        file_object = filedialog.askopenfile(
            mode='r',
            filetypes=[("Graph files", "*.graph")]
        )
        if file_object is not None:
            adjacency_matrix = json.load(file_object)
            if not self.check_adjacency_matrix(adjacency_matrix):
                messagebox.showerror(title="Application Error", message="Bad file!")
                return
            self.destroy_grid()
            self.n = len(adjacency_matrix)
            self.build_grid()
            old_reset_algorithm = self.reset_algorithm
            self.reset_algorithm = lambda: None
            for i in range(self.n):
                for j in range(self.n):
                    self.vars[i][j].set(f"{abs(adjacency_matrix[i][j]):g}")
            self.reset_algorithm = old_reset_algorithm
            self.reset_algorithm()
            file_object.close()

    def show_graph(self):
        self.ax.clear()
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
        if self.best_individuals and 0 <= self.current_iteration < len(self.best_individuals):
            individual = self.best_individuals[self.current_iteration]
            node_colors = ['red' if individual[i] == 1 else 'lightblue' for i in range(self.n)]
            nx.draw(G, ax=self.ax, with_labels=True, node_color=node_colors, edge_color='gray')
        else:
            nx.draw(G, ax=self.ax, with_labels=True, node_color='lightblue', edge_color='gray')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmApp(root)
    root.mainloop()
