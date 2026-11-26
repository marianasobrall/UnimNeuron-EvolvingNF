# Ponto 3 — Memória das regras (T29)

## O que foi medido

Usámos os registos passivos produzidos pela experiência T16 em `experiments/drift_agrawal.py`. Cada regra tem um ID permanente. Reconstituímos, **após o treino de cada observação de controlo**, o seu centro (vetor de 9 dimensões), sigma e suporte. O suporte é a contagem interna do modelo, atualizada por adaptações e fusões; não é o número de previsões feitas pela regra.

Os registos T16 de previsões permitem contar quantas vezes cada regra teve a **maior ativação antes da atualização**, separadamente antes do drift (1–1999), durante o intervalo de drift (2000–3000) e depois (3001–20000). Como esses registos guardam apenas a vencedora, repetimos o mesmo stream com o modelo instrumentado para resumir também a ativação **de todas as regras ativas**, sem alterar o classificador. Em cada fase guardamos, por ID, a quantidade de observações em que a regra existia, a ativação bruta média, a ativação relativa média (ativação dividida pela soma das ativações na observação) e a fração de vitórias condicionada à presença da regra. A ativação relativa é o peso anterior às probabilidades de classe da regra; não é, isoladamente, a sua contribuição para a classe prevista.

Para a mesma regra presente em dois momentos, calculámos a distância euclidiana entre os centros, a razão `sigma_depois / sigma_antes` e a diferença de suporte. Regras novas e desaparecidas são contadas em separado; não se compara o centro de duas regras com IDs diferentes.

## Como reproduzir

Na raiz do repositório, depois de gerar os dados T16 (`python experiments/drift_agrawal.py`), correr:

```bash
python experiments/rule_memory.py
python experiments/rule_activation_trace.py
python -m unittest discover -s tests -v
```

Os resultados ficam em `results/4_rule_memory/`: snapshots de 250 em 250 observações (com o ponto adicional 1999), comparações 1999→3000 e 1999→3500, tabelas de ativações por fase, dois resumos JSON e dois gráficos. `rule_memory.py` reconstitui os parâmetros diretamente dos registos T16; `rule_activation_trace.py` repete a experiência para medir todas as ativações. Esta repetição confirma, observação a observação, os rótulos, as previsões, o número de regras e os IDs das regras dominantes que constam dos ficheiros T16. O primeiro script verifica que o número final de regras reconstituído coincide com a experiência T16.

## Leitura preliminar desta experiência

| Configuração | Regras ativas em 1999 | Regras ativas em 3000 | Regras de 1999 ainda presentes | Novas regras ativas | Mediana da deslocação do centro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `max_rules=5` | 5 | 5 | 5 | 0 | 0,0348 |
| Sem limite | 38 | 46 | 38 | 8 | 0,0700 |

As cinco regras ativas no momento 1999 continuam ativas no momento 3000 na configuração limitada. **Isto não significa que o modelo não tenha tentado criar outras regras**: o T16 regista muitos candidatos criados e imediatamente removidos quando o limite é atingido. No modelo sem limite, as 38 regras iniciais continuam ativas e há oito regras adicionais no fim do intervalo de drift. A mediana da razão entre os sigmas é aproximadamente 0,9996 em ambas as configurações, ou seja, o sigma dessas regras alterou-se pouco neste intervalo. O centro moveu-se mais, em mediana, no modelo sem limite.

As ativações completas mostram outra cautela importante: no modelo limitado, a entropia normalizada média das ativações é aproximadamente **0,99995** antes, durante e depois do drift (1 significaria pesos iguais entre as regras). Sem limite, fica entre **0,993 e 0,996**. Portanto, muitas regras recebem simultaneamente pesos parecidos. A regra vencedora pode mudar mesmo quando as restantes ainda estão bastante ativadas. Além disso, a ativação relativa de cada regra tende a diminuir quando o número de regras cresce; essa diminuição, por si só, não demonstra perda de memória. Os ficheiros `*_all_activations.csv` mostram também a ativação bruta e o número de observações em que cada regra existiu, para permitir esta distinção.

Este resultado mede mudança na representação e uso das regras, **não prova retenção nem esquecimento de um conceito**. Para testar recuperação de memória é necessário um stream recorrente A–B–A, comparar as ativações das *mesmas* regras quando A regressa e avaliar o desempenho prequential. A fase “durante o drift” mistura gradualmente os conceitos e não deve ser tratada como um conceito puro. As fases têm comprimentos diferentes: por isso, as tabelas distinguem frações calculadas sobre a fase inteira das frações calculadas apenas enquanto a regra estava presente.

O pré-processamento e os parâmetros são os da experiência T16; em particular, ela usa min–max calculado sobre o stream completo. Esta análise descritiva herda essa escolha. Uma avaliação final estritamente online deve discutir ou substituir esse passo por normalização causal e aplicar a mesma regra a todas as configurações comparadas.
