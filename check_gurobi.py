"""
check_gurobi.py
===============
Standalone utility to verify that the Gurobi licence is correctly
installed and activated. Run this before executing the pipeline if
you encounter solver errors.

It tests problem scale by creating 2,500 binary variables — the size
required by the full simulated universe. Full licences have no limit;
restricted/trial licences cap at 2,000 variables.

Usage:
    python check_gurobi.py
"""

import gurobipy as gp

env = gp.Env()
m = gp.Model(env=env)

print("Gurobi version:", gp.gurobi.version())

# Attempt to create 2500 binary variables — full licences have no limit,
# but restricted/trial licences cap at 2000 variables.
x = m.addVars(2500, vtype=gp.GRB.BINARY, name="x")
m.update()
print(f"Variables created: {m.NumVars}")
print("No variable limit on your licence!")
