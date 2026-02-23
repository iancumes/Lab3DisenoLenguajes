#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regex_dfa.py

Construcción directa de AFD (DFA) a partir de una Expresión Regular (RE)
usando el método del árbol sintáctico con funciones:
nullable, firstpos, lastpos, followpos (Aho–Sethi–Ullman / "direct DFA construction").

Incluye:
- Shunting-yard (infix -> postfix) con concatenación explícita
- Construcción de árbol sintáctico desde postfix
- Construcción directa de DFA
- Simulación DFA (aceptación de cadenas)
- Minimización DFA (Hopcroft)
- Exportación a Graphviz DOT (visualización)

NOTAS DE REFERENCIA (para tu README):
- Construcción directa con followpos: Aho, Sethi, Ullman (Compilers) / Dragon Book.
- Shunting Yard: Edsger Dijkstra.
- Minimización: Hopcroft (partition refinement).

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, FrozenSet, List, Tuple, Optional, Iterable
import argparse
import sys
import string


# -------------------------
# Tokenización + concatenación explícita
# -------------------------

# Operadores soportados:
#   unión: |
#   concatenación: .   (se inserta automáticamente)
#   cerradura: *  (Kleene)
#   más: +        (1 o más)
#   opcional: ?   (0 o 1)
#   paréntesis: ( )
#
# Símbolos:
#   - cualquier caracter literal (incluye dígitos y letras)
#   - escapado con \  (p.ej. \| \* \( \) \. \\)
#   - epsilon opcional: ε  (U+03B5) o la palabra "eps" (tratada como ε)

EPSILON = "ε"
ENDMARK = "#"  # marcador de fin para construcción directa


def _is_literal(tok: str) -> bool:
    """True si tok es símbolo (hoja) y no operador/paréntesis."""
    return tok not in {"|", ".", "*", "+", "?", "(", ")"}


def tokenize(regex: str) -> List[str]:
    """Convierte string regex a lista de tokens (literals y operadores)."""
    tokens: List[str] = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            if i + 1 >= len(regex):
                raise ValueError("Escape '\\' al final de la expresión.")
            tokens.append(regex[i + 1])
            i += 2
            continue
        # palabra eps
        if regex.startswith("eps", i):
            tokens.append(EPSILON)
            i += 3
            continue
        if c in {"|", "*", "+", "?", "(", ")"}:
            tokens.append(c)
            i += 1
            continue
        # cualquier otro char literal
        tokens.append(c)
        i += 1
    return tokens


def insert_concatenation(tokens: List[str]) -> List[str]:
    """
    Inserta '.' cuando hay concatenación implícita.
    Regla típica: si (A)(B) donde A puede terminar expresión y B puede iniciar expresión.
    """
    out: List[str] = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i == len(tokens) - 1:
            break
        a = t
        b = tokens[i + 1]

        # a puede "terminar" un factor si es:
        # literal, ')', '*', '+', '?'
        a_can_end = _is_literal(a) or a in {")", "*", "+", "?"}
        # b puede "iniciar" un factor si es:
        # literal, '(', EPSILON
        b_can_start = _is_literal(b) or b == "("

        if a_can_end and b_can_start:
            out.append(".")
    return out


# -------------------------
# Shunting Yard (infix -> postfix)
# -------------------------

PRECEDENCE = {
    "|": 1,
    ".": 2,
    "*": 3,
    "+": 3,
    "?": 3,
}

RIGHT_ASSOCIATIVE = {"*", "+", "?"}  # unarios posfijos


def to_postfix(tokens: List[str]) -> List[str]:
    """
    Shunting-yard para operadores: |, ., *, +, ? y paréntesis.
    Unarios (* + ?) son posfijos: en la práctica, al leerlos se manejan como operadores.
    """
    output: List[str] = []
    stack: List[str] = []

    for tok in tokens:
        if _is_literal(tok):
            output.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Paréntesis desbalanceados: falta '('")
            stack.pop()  # quita '('
        else:
            # operador
            while stack:
                top = stack[-1]
                if top == "(":
                    break
                if (PRECEDENCE[top] > PRECEDENCE[tok]) or (
                    PRECEDENCE[top] == PRECEDENCE[tok]
                    and tok not in RIGHT_ASSOCIATIVE
                ):
                    output.append(stack.pop())
                else:
                    break
            stack.append(tok)

    while stack:
        if stack[-1] in {"(", ")"}:
            raise ValueError("Paréntesis desbalanceados.")
        output.append(stack.pop())

    return output


# -------------------------
# Árbol sintáctico (postfix -> AST)
# -------------------------

@dataclass(frozen=True)
class Node:
    typ: str  # 'leaf', 'eps', 'or', 'cat', 'star', 'plus', 'opt'
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    sym: Optional[str] = None
    pos: Optional[int] = None  # solo para hojas con posición


def build_syntax_tree(postfix: List[str]) -> Tuple[Node, Dict[int, str], int]:
    """
    Construye AST desde postfix.
    Devuelve: (raíz, pos_to_symbol, end_pos)
    """
    stack: List[Node] = []
    pos_to_sym: Dict[int, str] = {}
    next_pos = 1

    def new_leaf(symbol: str) -> Node:
        nonlocal next_pos
        if symbol == EPSILON:
            return Node(typ="eps")
        p = next_pos
        next_pos += 1
        pos_to_sym[p] = symbol
        return Node(typ="leaf", sym=symbol, pos=p)

    for tok in postfix:
        if _is_literal(tok):
            stack.append(new_leaf(tok))
        elif tok in {"*", "+", "?"}:
            if not stack:
                raise ValueError(f"Operador unario '{tok}' sin operando.")
            a = stack.pop()
            if tok == "*":
                stack.append(Node(typ="star", left=a))
            elif tok == "+":
                stack.append(Node(typ="plus", left=a))
            else:
                stack.append(Node(typ="opt", left=a))
        elif tok in {"|", "."}:
            if len(stack) < 2:
                raise ValueError(f"Operador binario '{tok}' sin suficientes operandos.")
            b = stack.pop()
            a = stack.pop()
            if tok == "|":
                stack.append(Node(typ="or", left=a, right=b))
            else:
                stack.append(Node(typ="cat", left=a, right=b))
        else:
            raise ValueError(f"Token desconocido en postfix: {tok!r}")

    if len(stack) != 1:
        raise ValueError("Expresión postfix inválida (sobran operandos).")

    root = stack[0]

    # Aumentar RE con marcador de fin: (root . #)
    end_leaf = new_leaf(ENDMARK)
    end_pos = end_leaf.pos  # type: ignore
    augmented = Node(typ="cat", left=root, right=end_leaf)

    return augmented, pos_to_sym, end_pos


# -------------------------
# Funciones nullable/firstpos/lastpos/followpos  (GEnerado con ChatGPT: Se le adjunto una imagen de las notas de clase con una instruccion de lo que consistia cada una de las funciones calculadas con ejemplos.)
# -------------------------

@dataclass
class PosInfo:
    nullable: bool
    firstpos: Set[int]
    lastpos: Set[int]


def compute_functions(root: Node, followpos: Dict[int, Set[int]]) -> PosInfo:
    """
    Post-orden para calcular nullable, firstpos, lastpos y llenar followpos.
    followpos se modifica in-place.
    """
    def rec(n: Node) -> PosInfo:
        if n.typ == "eps":
            return PosInfo(nullable=True, firstpos=set(), lastpos=set())
        if n.typ == "leaf":
            assert n.pos is not None
            followpos.setdefault(n.pos, set())
            return PosInfo(nullable=False, firstpos={n.pos}, lastpos={n.pos})

        if n.typ == "or":
            A = rec(n.left)   # type: ignore
            B = rec(n.right)  # type: ignore
            return PosInfo(
                nullable=A.nullable or B.nullable,
                firstpos=A.firstpos | B.firstpos,
                lastpos=A.lastpos | B.lastpos,
            )

        if n.typ == "cat":
            A = rec(n.left)   # type: ignore
            B = rec(n.right)  # type: ignore

            # followpos: para cada i en lastpos(A), followpos(i) incluye firstpos(B)
            for i in A.lastpos:
                followpos.setdefault(i, set()).update(B.firstpos)

            first = (A.firstpos | B.firstpos) if A.nullable else set(A.firstpos)
            last = (A.lastpos | B.lastpos) if B.nullable else set(B.lastpos)
            return PosInfo(
                nullable=A.nullable and B.nullable,
                firstpos=first,
                lastpos=last,
            )

        if n.typ == "star":
            A = rec(n.left)  # type: ignore
            for i in A.lastpos:
                followpos.setdefault(i, set()).update(A.firstpos)
            return PosInfo(nullable=True, firstpos=set(A.firstpos), lastpos=set(A.lastpos))

        if n.typ == "plus":
            A = rec(n.left)  # type: ignore
            for i in A.lastpos:
                followpos.setdefault(i, set()).update(A.firstpos)
            # nullable(plus) = nullable(A) ? (si A puede ser ε, entonces 1+ aún requiere 1 ocurrencia,
            # pero si A puede ser ε, entonces sí puede producir ε con esa única ocurrencia).
            # En el formalismo clásico: nullable(A+) = nullable(A)
            return PosInfo(nullable=A.nullable, firstpos=set(A.firstpos), lastpos=set(A.lastpos))

        if n.typ == "opt":
            A = rec(n.left)  # type: ignore
            return PosInfo(nullable=True, firstpos=set(A.firstpos), lastpos=set(A.lastpos))

        raise ValueError(f"Tipo de nodo desconocido: {n.typ}")

    return rec(root)


# -------------------------
# DFA (construcción directa)
# -------------------------

@dataclass
class DFA:
    states: List[FrozenSet[int]]
    start: FrozenSet[int]
    accept: Set[FrozenSet[int]]
    alphabet: List[str]
    trans: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]]

    def step(self, state: FrozenSet[int], ch: str) -> FrozenSet[int]:
        return self.trans.get((state, ch), frozenset())

    def simulate(self, w: str) -> bool:
        s = self.start
        for ch in w:
            s = self.step(s, ch)
        return s in self.accept


def build_dfa(root: Node, pos_to_sym: Dict[int, str], end_pos: int, followpos: Dict[int, Set[int]]) -> DFA:
    # alfabeto = símbolos excepto endmarker y epsilon
    alpha = sorted({sym for sym in pos_to_sym.values() if sym not in {ENDMARK, EPSILON}})

    # start state
    info = compute_functions(root, followpos)
    start = frozenset(info.firstpos)

    states: List[FrozenSet[int]] = []
    accept: Set[FrozenSet[int]] = set()
    trans: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]] = {}

    # BFS sobre estados
    unmarked: List[FrozenSet[int]] = [start]
    seen: Set[FrozenSet[int]] = {start}

    while unmarked:
        S = unmarked.pop(0)
        states.append(S)

        if end_pos in S:
            accept.add(S)

        for a in alpha:
            U: Set[int] = set()
            for p in S:
                if pos_to_sym[p] == a:
                    U.update(followpos.get(p, set()))
            U_f = frozenset(U)
            trans[(S, a)] = U_f
            if U_f not in seen:
                seen.add(U_f)
                unmarked.append(U_f)

    return DFA(states=states, start=start, accept=accept, alphabet=alpha, trans=trans)


# -------------------------
# Minimización DFA (Hopcroft) (Reutilizado de codigos de Teoria de la Computacion)
# -------------------------

def minimize_dfa(dfa: DFA) -> DFA:
    # Remover estado muerto implícito si no aparece en states: aseguramos que exista si es destino
    all_states: Set[FrozenSet[int]] = set(dfa.states)
    for (_, _a), dst in list(dfa.trans.items()):
        all_states.add(dst)

    # Partición inicial: F y Q\F
    F = set(dfa.accept)
    Q = set(all_states)
    NF = Q - F

    P: List[Set[FrozenSet[int]]] = []
    if F:
        P.append(set(F))
    if NF:
        P.append(set(NF))

    W: List[Set[FrozenSet[int]]] = []
    if F:
        W.append(set(F))
    elif NF:
        W.append(set(NF))

    # Precompute inverse transitions: inv[a][q] = {p | δ(p,a)=q}
    inv: Dict[str, Dict[FrozenSet[int], Set[FrozenSet[int]]]] = {a: {} for a in dfa.alphabet}
    for (p, a), q in dfa.trans.items():
        inv[a].setdefault(q, set()).add(p)

    while W:
        A = W.pop()
        for c in dfa.alphabet:
            # X = { q | δ(q,c) in A }
            X: Set[FrozenSet[int]] = set()
            for q in A:
                X |= inv[c].get(q, set())

            if not X:
                continue

            newP: List[Set[FrozenSet[int]]] = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.append(inter)
                    newP.append(diff)
                    # mantener W consistente
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        # añadir el más pequeño
                        if len(inter) <= len(diff):
                            W.append(inter)
                        else:
                            W.append(diff)
                else:
                    newP.append(Y)
            P = newP

    # Mapear cada estado a su bloque representativo
    block_of: Dict[FrozenSet[int], int] = {}
    blocks: List[Set[FrozenSet[int]]] = []
    for i, block in enumerate(P):
        blocks.append(block)
        for q in block:
            block_of[q] = i

    # Construir DFA minimizado
    new_states: List[FrozenSet[int]] = []
    rep_state: List[FrozenSet[int]] = []
    for block in blocks:
        rep = next(iter(block))
        rep_state.append(rep)
        # representamos el estado como el conjunto de posiciones del representante (solo para imprimir/ DOT)
        new_states.append(rep)

    new_start = rep_state[block_of[dfa.start]]
    new_accept: Set[FrozenSet[int]] = set()
    for block in blocks:
        rep = next(iter(block))
        if any(q in dfa.accept for q in block):
            new_accept.add(rep)

    new_trans: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]] = {}
    for block in blocks:
        rep = next(iter(block))
        for a in dfa.alphabet:
            dst = dfa.trans.get((rep, a), frozenset())
            dst_rep = rep_state[block_of[dst]]
            new_trans[(rep, a)] = dst_rep

    return DFA(
        states=new_states,
        start=new_start,
        accept=new_accept,
        alphabet=list(dfa.alphabet),
        trans=new_trans,
    )


# -------------------------
# DOT (Graphviz)
# -------------------------
#Funcion para graficar el DFA usando Graphviz DOT. Prompt: Podrias a este codigo (Todo el codigo de la generacion del afd) agregarle una generacion del AFd utilizando la herramienta Graphviz.

def dfa_to_dot(dfa: DFA, title: str = "DFA") -> str:
    def sid(s: FrozenSet[int]) -> str:
        # id estable
        return "S" + str(abs(hash(s)) % 10_000_000)

    def label(s: FrozenSet[int]) -> str:
        if not s:
            return "∅"
        return "{" + ",".join(map(str, sorted(s))) + "}"

    lines = []
    lines.append("digraph DFA {")
    lines.append('  rankdir=LR;')
    lines.append(f'  labelloc="t"; label="{title}"; fontsize=20;')
    lines.append("  node [shape=circle];")
    lines.append("  __start [shape=none label=\"\"];")
    lines.append(f"  __start -> {sid(dfa.start)};")

    # estados
    for s in set(dfa.states) | {dst for (_, _), dst in dfa.trans.items()}:
        shape = "doublecircle" if s in dfa.accept else "circle"
        lines.append(f'  {sid(s)} [shape={shape} label="{label(s)}"];')

    # agrupar etiquetas por (src,dst)
    edge_labels: Dict[Tuple[FrozenSet[int], FrozenSet[int]], List[str]] = {}
    for (src, a), dst in dfa.trans.items():
        edge_labels.setdefault((src, dst), []).append(a)

    for (src, dst), lbls in edge_labels.items():
        text = ",".join(lbls)
        lines.append(f'  {sid(src)} -> {sid(dst)} [label="{text}"];')

    lines.append("}")
    return "\n".join(lines)


# -------------------------
# Pretty print
# -------------------------

def print_dfa(dfa: DFA) -> None:
    states_set = set(dfa.states) | {dst for (_, _), dst in dfa.trans.items()}
    # orden aproximado por tamaño y contenido
    ordered = sorted(states_set, key=lambda s: (len(s), sorted(s)))
    idx = {s: i for i, s in enumerate(ordered)}

    print("\n== DFA ==")
    print(f"Alfabeto: {dfa.alphabet}")
    print(f"Estado inicial: q{idx[dfa.start]} = {sorted(dfa.start)}")
    print("Estados de aceptación:", ", ".join([f"q{idx[s]}" for s in ordered if s in dfa.accept]) or "(ninguno)")
    print("\nTransiciones:")
    for s in ordered:
        for a in dfa.alphabet:
            dst = dfa.trans.get((s, a), frozenset())
            print(f"  δ(q{idx[s]}, {a!r}) = q{idx[dst]}")

    print("\nLeyenda estados:")
    for s in ordered:
        mark = " (start)" if s == dfa.start else ""
        mark += " (accept)" if s in dfa.accept else ""
        print(f"  q{idx[s]} = {sorted(s)}{mark}")


# -------------------------
# Pipeline completo
# -------------------------

def compile_regex_to_dfa(regex: str, minimize: bool = False) -> DFA:
    toks = tokenize(regex)
    toks = insert_concatenation(toks)
    postfix = to_postfix(toks)
    root, pos_to_sym, end_pos = build_syntax_tree(postfix)
    followpos: Dict[int, Set[int]] = {}
    dfa = build_dfa(root, pos_to_sym, end_pos, followpos)
    if minimize:
        dfa = minimize_dfa(dfa)
    return dfa


# -------------------------
# CLI / REPL
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Construcción directa de DFA desde una expresión regular.")
    ap.add_argument("--regex", "-r", required=True, help="Expresión regular en infix. Ej: a(b|c)*")
    ap.add_argument("--min", action="store_true", help="Minimizar DFA (Hopcroft).")
    ap.add_argument("--dot", help="Guardar archivo DOT (Graphviz) del DFA resultante.")
    ap.add_argument("--dot-title", default="DFA", help="Título del DOT.")
    ap.add_argument("--no-print", action="store_true", help="No imprimir tabla del DFA.")
    ap.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo para probar cadenas.")
    ap.add_argument("--test", nargs="*", help="Cadenas a evaluar (si no usas -i).")
    args = ap.parse_args()

    try:
        dfa = compile_regex_to_dfa(args.regex, minimize=args.min)
    except Exception as e:
        print(f"Error compilando RE: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.no_print:
        print_dfa(dfa)

    if args.dot:
        dot = dfa_to_dot(dfa, title=args.dot_title)
        with open(args.dot, "w", encoding="utf-8") as f:
            f.write(dot)
        print(f"\nDOT guardado en: {args.dot}")
        print("Tip: si tienes Graphviz instalado, puedes renderizar:")
        print(f"  dot -Tpng {args.dot} -o dfa.png")

    def eval_string(w: str):
        ok = dfa.simulate(w)
        print(f"{w!r} -> {'ACEPTADA' if ok else 'RECHAZADA'}")

    if args.interactive:
        print("\nModo interactivo. Escribe cadenas (enter vacío para salir).")
        while True:
            try:
                w = input("> ")
            except EOFError:
                break
            if w == "":
                break
            eval_string(w)
    else:
        if args.test is not None:
            for w in args.test:
                eval_string(w)


if __name__ == "__main__":
    main()