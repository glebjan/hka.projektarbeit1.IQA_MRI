# Logical view

Sources: `logical-view.mmd` (class diagram), `dataflow.mmd` (flowchart). Rendered SVGs sit beside them and are copied into the report.

Render with the config file, otherwise Mermaid 11 emits `<foreignObject>` HTML labels that Typst draws as empty boxes:

```bash
mmdc -c mermaid.config.json -i logical-view.mmd -o logical-view.svg -b transparent
mmdc -c mermaid.config.json -i dataflow.mmd -o dataflow.svg -b transparent
```
