# FAQ – Help & Guidance

---

### How do I choose the number of initial experiments (`n_init`) for Bayesian Optimization?
A common rule is to set  
**`n_init = 2 to 3 × number of variables`**.  
This helps the surrogate model (e.g., Gaussian Process) build a good initial understanding of the system.

---

### When should I stop a Bayesian Optimization campaign if I’m not using a fixed number of iterations?
You can stop when recent experiments no longer significantly improve the result (**convergence**),  
or when the **acquisition function** suggests very low improvement potential.

---

### What do the acquisition functions (EI, PI, LCB) mean, and how do I pick one?
- **EI (Expected Improvement):** balances exploration and exploitation — a safe default.  
- **PI (Probability of Improvement):** more conservative, favors steady progress.  
- **LCB (Lower Confidence Bound):** more exploratory, searches unexplored regions.

---

### What does the “Yield” graph show, and why is it useful during optimization?
The **Yield** graph shows the **best result found over time**.  
It helps you track progress, detect plateaus, and decide when the optimization might be complete.

---

### How is the “next suggestion” generated, and what data is used to make it?
BO studio uses the **surrogate model** (built from your past data)  
and the **acquisition function** to choose the next experimental condition with the highest potential.

---

### How do I export my optimization results, and what formats are supported?
You can export results as a **CSV file** after the campaign ends or you can store them in the **database**.

---

### Is it possible to optimize more than one objective (e.g., Yield and Productivity)?
Yes! BO studio supports **multi-objective optimization**.  
You can select more than one target and explore trade-offs using **Pareto front analysis**.

---
