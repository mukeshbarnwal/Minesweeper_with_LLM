
**Minesweeper LLM Agent**
A simulation framework where a LLM plays the Minesweeper game intelligently by combining deterministic logic, probability heuristics, and tie-breaking via LLM reasoning.

#Setup

1. **Clone the repository in your local**

   ```bash
   git clone https://github.com/mukeshbarnwal/Minesweeper_with_LLM.git
   cd Minesweeper_with_LLM
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example requirements:

   ```txt
   openai
   pandas
   tqdm
   ```

3. **Set your OpenAI API Key/Local Model**

   * Export it as an environment variable:

     ```bash
     export OPENAI_API_KEY=your-key-here
     ```
   * Or modify the code to use a local model like LLaMA via an HTTP endpoint. For this we have to download ollama and then pull models like Llama3:8b

4. Run 2 Models:
   1. **External (gpt 4o)**: **Run the LLM_minesweeper_notebook.ipynb**
   2. **Local(Llama3:8b):** **Run the LLM_Minesweeper_Local_Model_Llama3_8b**
6. Results for the above 2 models:
   1. Overall results shown in the notebooks while it ran for the first time
   2. Individual game results:
       1. External model result: **results.csv**
       2. Local model result: **results_llama3_8b.csv**



## Architecture Overview

* **MinesweeperBoard**
  This is the main game engine. It creates the board, places the mines, and handles how the cells open.

* **deterministic\_step()**
  This function applies standard Minesweeper logic. For example:

  * If a number equals the number of flags around it, the rest of the nearby hidden cells are safe to open.
  * If the number minus the nearby flags equals the number of hidden cells, then all those hidden cells must be mines, so it flags them.

* **compute\_probabilities()**
  When logic isn't enough, this function calculates how likely each unknown cell is to have a mine based on nearby clues.

* **query\_llm()**
  If several cells look equally safe, this function asks the language model (LLM) to pick one of them to open.

* **play\_one()**
  Runs one full game using the above logic, probability, and LLM decision-making.

* **run\_experiment()**
  Runs multiple games in a row, saves the results in a CSV file, and shows a summary of how the bot performed.


## Prompt Strategy

When there's a tie among cells with the lowest mine probability:

* A full board snapshot and a probability table of tied candidates is sent to the LLM.
* The prompt instructs the LLM to return a JSON dict like:

  ```json
  {"row": 4, "col": 6, "action": "open"}
  ```
* This uses `gpt-4o-mini` by default, but can be adapted for local inference.

---

## Output

1. External model:
A `results.csv` is created/updated with:

* Game ID
* Board size and mine count
* Total moves
* Win/loss status
* Runtime
* RNG seed

2. Local model:
A `results_llama3_8b.csv` is created/updated with:

* Game ID
* Board size and mine count
* Total moves
* Win/loss status
* Runtime
* RNG seed

  
A summary is printed after N games for the both models:

```
Batch summary:
outcome
loss     3
win     17
dtype: int64
Mean moves: 18.3
```

---

## Heuristics & Design Choices

* **Hybrid reasoning**: Classical logic handles easy moves. Probability estimation follows. LLM only steps in when probabilities tie.
* **Seeded randomness**: Each game uses a consistent RNG seed for reproducibility.
* **Efficiency**: Probability heuristics avoid full CSP/MC sampling (though upgradeable).
* **Model fallback**: If LLM fails to return valid JSON, fallback is to open the first best cell.

---

## Known Limitations

* No deep constraint solver or full Monte Carlo sampling.
* LLM intervention increases API cost and latency (if using OpenAI).
* Assumes symmetric flagging/opening confidence; may fail on more complex boards.
* Not trained end-to-end; LLM is not "learning" from wins/losses.
