import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🏗️ SH TechForge Precast Yard Optimizer")

# Dummy Data
np.random.seed(42)
products = ["Wall Panel", "Beam", "Column"]
beds = ["Bed 1", "Bed 2", "Bed 3"]

data = pd.DataFrame({
    "Product": products,
    "Base_Cycle_Time_hr": np.random.randint(10, 24, size=3),
    "Labour_req": np.random.randint(4, 10, size=3),
    "Profit_per_unit": np.random.randint(5000, 12000, size=3),
    "Demand_limit": np.random.randint(20, 50, size=3),
    "Mould_required": np.random.randint(1, 3, size=3)
})

# Sidebar Inputs
st.sidebar.header("🔧 Yard Inputs")
total_labour = st.sidebar.slider("Total Labour", 20, 200, 100)
total_moulds = st.sidebar.slider("Total Moulds", 10, 100, 40)
curing_type = st.sidebar.selectbox("Curing Type", ["Normal Curing", "Steam Curing"])
region_temp = st.sidebar.slider("Temperature (°C)", 5, 45, 25)

# Strength Model
time_hours = np.array([6, 12, 18, 24, 30, 36]).reshape(-1, 1)
strength = np.array([15, 22, 30, 38, 42, 48])

model = LinearRegression()
model.fit(time_hours, strength)

target_strength = 35
predicted_time = (target_strength - model.intercept_) / model.coef_[0]

temp_factor = 1 - (region_temp - 25) * 0.01
predicted_time *= temp_factor

if curing_type == "Steam Curing":
    predicted_time *= 0.75
    extra_cost = 50000
else:
    extra_cost = 0

data["Effective_Cycle_Time"] = np.maximum(data["Base_Cycle_Time_hr"], predicted_time)

# LPP Optimization
model_lp = LpProblem(name="Precast_Optimization", sense=LpMaximize)
x = {i: LpVariable(name=i, lowBound=0, cat='Integer') for i in products}

model_lp += lpSum(data.loc[data["Product"] == i, "Profit_per_unit"].values[0] * x[i]
                  for i in products)

model_lp += lpSum(data.loc[data["Product"] == i, "Labour_req"].values[0] * x[i]
                  for i in products) <= total_labour

model_lp += lpSum(data.loc[data["Product"] == i, "Mould_required"].values[0] * x[i]
                  for i in products) <= total_moulds

for i in products:
    demand = data.loc[data["Product"] == i, "Demand_limit"].values[0]
    model_lp += x[i] <= demand

model_lp.solve()

production_plan = {i: int(x[i].value()) for i in products}

# KPI Calculations
used_labour = sum(
    data.loc[data["Product"] == i, "Labour_req"].values[0] * production_plan[i]
    for i in products
)

used_moulds = sum(
    data.loc[data["Product"] == i, "Mould_required"].values[0] * production_plan[i]
    for i in products
)

labour_util = (used_labour / total_labour) * 100 if total_labour else 0
mould_util = (used_moulds / total_moulds) * 100 if total_moulds else 0

total_profit = sum(
    production_plan[i] * data.loc[data["Product"] == i, "Profit_per_unit"].values[0]
    for i in products
) - extra_cost

total_time = sum(
    production_plan[i] * data.loc[data["Product"] == i, "Effective_Cycle_Time"].values[0]
    for i in products
)

baseline_time = sum(
    data["Base_Cycle_Time_hr"] * data["Demand_limit"]
)

cycle_reduction = ((baseline_time - total_time) / baseline_time) * 100

efficiency_score = (labour_util + mould_util + cycle_reduction) / 3

# KPI Dashboard
st.subheader("📊 KPI Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("⏱️ Cycle Time Reduction", f"{cycle_reduction:.1f}%")
col2.metric("👷 Labour Utilization", f"{labour_util:.1f}%")
col3.metric("🧱 Mould Utilization", f"{mould_util:.1f}%")
col4.metric("🏭 Yard Efficiency Score", f"{efficiency_score:.1f}%")

# Production Plan
st.subheader("📦 Optimal Production Plan")
st.table(pd.DataFrame(list(production_plan.items()), columns=["Product", "Quantity"]))

st.write(f"💰 **Total Profit:** ₹{int(total_profit)}")
st.write(f"⏱️ **Total Cycle Time:** {total_time:.2f} hours")

# Transportation Model
st.subheader("🚚 Casting Bed Allocation")

bed_capacity = {"Bed 1": 40, "Bed 2": 35, "Bed 3": 30}

cost_matrix = pd.DataFrame(
    np.random.randint(5, 20, size=(3, 3)),
    index=products,
    columns=beds
)

st.dataframe(cost_matrix)

transport_model = LpProblem("Bed_Allocation", LpMinimize)

y = LpVariable.dicts("assign",
                     [(p, b) for p in products for b in beds],
                     lowBound=0,
                     cat='Integer')

transport_model += lpSum(cost_matrix.loc[p, b] * y[(p, b)]
                         for p in products for b in beds)

for p in products:
    transport_model += lpSum(y[(p, b)] for b in beds) == production_plan[p]

for b in beds:
    transport_model += lpSum(y[(p, b)] for p in products) <= bed_capacity[b]

transport_model.solve()

allocation = []
for p in products:
    for b in beds:
        qty = int(y[(p, b)].value())
        if qty > 0:
            allocation.append([p, b, qty])

st.table(pd.DataFrame(allocation, columns=["Product", "Bed", "Qty"]))

# Cost vs Time Graph
st.subheader("📉 Cost vs Time Trade-off")

normal_time = total_time
steam_time = total_time * 0.75

fig, ax = plt.subplots()
ax.plot(["Normal", "Steam"], [normal_time, steam_time])
ax.set_ylabel("Cycle Time (hrs)")
ax.set_title("Curing Strategy Comparison")
st.pyplot(fig)