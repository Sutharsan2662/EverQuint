import streamlit as st

# Page setup
st.set_page_config(page_title="Max Profit Calculator", layout="wide")

# Buildings configuration
buildings = {
    "T": {"name": "Theatre", "build": 5, "earn": 1500, "icon": "🎭"},
    "P": {"name": "Pub", "build": 4, "earn": 1000, "icon": "🍺"},
    "C": {"name": "Commercial Park", "build": 10, "earn": 2000, "icon": "🏢"}
}

# Correct max profit logic with accurate building counts
def max_profit_all(total_time):
    def solve(time_passed, built_counts, current_earnings):
        max_profit = current_earnings * (total_time - time_passed)
        best_solutions = []

        # Check if any building can be built
        can_build_any = False
        for b_key, b in buildings.items():
            if time_passed + b["build"] <= total_time:
                can_build_any = True
                build_time = b["build"]
                profit_during_build = current_earnings * build_time

                new_counts = built_counts.copy()
                new_counts[b_key] += 1
                new_earnings = current_earnings + b["earn"]

                remaining_profit, solutions = solve(time_passed + build_time, new_counts, new_earnings)
                total_profit = profit_during_build + remaining_profit

                if total_profit > max_profit:
                    max_profit = total_profit
                    best_solutions = solutions
                elif total_profit == max_profit:
                    best_solutions.extend(solutions)

        # Leaf node: no more building can be built
        if not can_build_any:
            best_solutions = [(built_counts.copy(), max_profit)]

        return max_profit, best_solutions

    return solve(0, {"T": 0, "P": 0, "C": 0}, 0)

# Header
st.title("🏗️ Max Profit Calculator")
st.write("Optimize your building strategy for maximum profit.")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Properties display
    st.subheader("Available Properties")
    prop_cols = st.columns(len(buildings))
    for idx, (k, b) in enumerate(buildings.items()):
        with prop_cols[idx]:
            st.write(f"{b['icon']} **{b['name']}**")
            st.write(f"Build Time: {b['build']} units")
            st.write(f"Earnings/Unit: ${b['earn']}")

    # Time input
    st.subheader("Time Input")
    time_units = st.number_input("Total Time Units", min_value=1, value=15, step=1)

    # Rules section
    st.subheader("📋 Rules")
    st.write("✓ Only one building can be constructed at a time")
    st.write("✓ Earnings start immediately after a building is completed")
    st.write("✓ Existing buildings earn while new buildings are being constructed")
    st.write("✓ Multiple buildings can be constructed sequentially")

with col2:
    # Calculate max profit
    profit, solutions = max_profit_all(time_units)

    # Filter unique solutions
    unique_solutions = []
    seen = set()
    for sol, p in solutions:
        key = (sol["T"], sol["P"], sol["C"])
        if key not in seen:
            seen.add(key)
            unique_solutions.append((sol, p))

    st.subheader("📊 Maximum Profit")
    st.write(f"**Profit:** ${profit}")

    if unique_solutions:
        st.markdown("### 💰 Optimal Building Strategies")

        # Columns for multiple solutions (up to 3 side-by-side)
        strategy_cols = st.columns(len(unique_solutions[:3]))
        for idx, (sol, _) in enumerate(unique_solutions[:3]):
            with strategy_cols[idx]:
                st.write(f"**Solution {idx+1}**")
                # Display exact number of each building built
                for b_key, count in sol.items():
                    st.write(f"{buildings[b_key]['icon']} {buildings[b_key]['name']}: {count}")
