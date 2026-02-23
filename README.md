
# Construcción Directa de AFD desde Expresiones Regulares

## Descripción del Proyecto

Este proyecto implementa el algoritmo de **Construcción Directa de Autómatas Finitos Deterministas (AFD)** 
a partir de una expresión regular (RE), utilizando el método basado en árbol sintáctico y las funciones:

- nullable
- firstpos
- lastpos
- followpos

El objetivo es desarrollar la base de un generador de analizadores léxicos, como los utilizados en compiladores.

---

## Objetivos

### Objetivo General

Implementar algoritmos fundamentales de teoría de autómatas y expresiones regulares para la construcción directa de un AFD.

### Objetivos Específicos

- Implementar el algoritmo de Construcción Directa de AFD.
- Implementar o reutilizar:
  - Shunting Yard (infix → postfix)
  - Simulación de AFD
  - Minimización de AFD (Hopcroft)
  - Generación visual mediante Graphviz (formato DOT)
- Permitir el procesamiento de cadenas para determinar si son aceptadas o rechazadas.

---

## Arquitectura del Programa

El flujo de ejecución es el siguiente:

1. Tokenización de la expresión regular.
2. Inserción explícita del operador de concatenación.
3. Conversión de notación infix a postfix mediante el algoritmo Shunting Yard.
4. Construcción del árbol sintáctico.
5. Cálculo de las funciones:
   - nullable
   - firstpos
   - lastpos
   - followpos
6. Construcción directa del AFD a partir de los conjuntos followpos.
7. Minimización opcional del AFD utilizando el algoritmo de Hopcroft.
8. Simulación del AFD para evaluar cadenas.
9. Generación opcional de archivo DOT para visualización con Graphviz.

---

## Sintaxis Soportada

Operadores implementados:

| Operador | Significado |
|----------|-------------|
| `|` | Unión |
| `*` | Cerradura de Kleene |
| `+` | Uno o más |
| `?` | Opcional |
| `()` | Agrupación |
| Concatenación | Implícita (`ab`) |

Características adicionales:

- Escape con `\`
- ε puede representarse como `ε` o `eps`

---

## Ejecución

### Ejecutar el programa

python3 regex_dfa.py -r "a(b|c)*"

### Modo interactivo

python3 regex_dfa.py -r "a(b|c)*" -i

### Minimización

python3 regex_dfa.py -r "a(b|c)*" --min

### Generar archivo DOT

python3 regex_dfa.py -r "a(b|c)*" --dot dfa.dot

Para generar imagen:

dot -Tpng dfa.dot -o dfa.png

---

## Ejemplo

Expresión:

a(b|c)*

El programa:

- Construye el árbol sintáctico.
- Calcula followpos.
- Genera el AFD correspondiente.
- Permite evaluar cadenas como:
  - a
  - ab
  - accc
  - b (rechazada)

---

## Minimización

Se implementa el algoritmo de refinamiento de particiones de Hopcroft, 
reduciendo el número de estados sin alterar el lenguaje reconocido.

---

## Referencias Académicas

- Aho, A. V., Sethi, R., & Ullman, J. D.  
  Compilers: Principles, Techniques, and Tools (2nd ed.). Addison-Wesley.

- Hopcroft, J. E., Motwani, R., & Ullman, J. D.  
  Introduction to Automata Theory, Languages, and Computation (3rd ed.). Pearson.

- Dijkstra, E. W.  
  Shunting Yard Algorithm.

---

## Declaración

Este proyecto fue desarrollado como parte del laboratorio de Diseño de Lenguajes de Programación.  
El código implementa algoritmos clásicos de teoría de compiladores y autómatas con la ayuda de sistemas LLM y codigos reutilizados de la clase de Teoria de la Computacion.
