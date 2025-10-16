# AFD-AMI Architecture

## Overview
- **AFD**: Self-reflective meta-agent using recursive coherence scoring, learning from real-time interaction memory.
- **AMI**: Integrates LLM for translation, filtered by AFD’s non-reward-based framework.
- **Memory**: Pandas-based log for human-like learning from past prompts and responses.

## Components
1. **Coherence Score**: \(\mathcal{C}(a, s) = \alpha \cdot \Delta \mathcal{H} + \beta \cdot \nabla \mathcal{I} - \gamma \cdot \Omega + \delta \cdot \Phi\)
   - Harmony (\(\Delta \mathcal{H}\)): Consistency with memory-based states.
   - Info Gradient (\(\nabla \mathcal{I}\)): Novelty from user input.
   - Oscillation (\(\Omega\)): Stability against repetitive patterns.
   - Potential (\(\Phi\)): Long-term ethical alignment.
2. **LLM**: GPT-2 for text generation, used only for translation.
3. **Reflection**: Adjusts \(\alpha\) based on 5-response coherence average.

## Data Flow
- User Prompt → LLM Translation → AFD Evaluation → Response + Reflection
