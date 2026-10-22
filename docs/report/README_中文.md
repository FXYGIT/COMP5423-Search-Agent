# COMP5423 Overleaf report scaffold

This project is based on the supplied ACM Overleaf template archive. The `acmart` document class and ACM bibliography style are retained; sample paper text and fake publication metadata have been replaced with a report outline for the COMP5423 Skill-Augmented Search Agent project.

## Files
- `main.tex`: report entry point and assignment-aligned section structure.
- `contributions.tex`: compulsory contribution/declaration section, with four editable member placeholders.
- `references.bib`: add only real references cited in the report.
- `acmart.cls`, ACM `.bst`/`.bbx`/`.cbx` and `.dbx` files: template support files.
- `figures/sign1.png`: retained from the provided archive, but the scaffold uses blank signature lines instead.

## Import into Overleaf
1. Download this ZIP file and sign in to Overleaf.
2. From the Overleaf dashboard, choose **New Project → Upload Project**.
3. Upload the ZIP file as-is. Overleaf should create a new project and unpack it.
4. Open `main.tex`. If Overleaf does not select it automatically, use the project menu to set `main.tex` as the main document.
5. Replace the four author placeholders near the top of `main.tex` with your team members' names.
6. Click **Recompile** to check the project. Fill sections as the implementation and experiments progress.

## Invite teammates
1. Open the project in Overleaf and click **Share** in the upper-right corner.
2. Invite each teammate using the email address associated with their Overleaf account, or use the available sharing-link option if your account offers it.
3. Choose the least-permissive role that fits the person: editing for co-authors, review/comment access for reviewers, or view-only for readers.
4. Ask each teammate to open the invitation and verify they can edit and recompile.
5. If Overleaf indicates your current plan limits the number of collaborators, check the current plan details or whether your university provides an institutional Overleaf subscription. Do not assume every plan permits the same number of editors.

## Assignment alignment checklist
- The body sections cover task/dataset, system design and workflow, retrieval/tools, search agent, reusable skills, UI, experimental setup, results, discussion, and conclusion.
- The workflow diagram and UI screenshot are clearly marked as scaffolds and must be updated to reflect the actual implementation.
- Replace every `[TODO]`/placeholder and every `--` result before final submission.
- Keep the contribution and declaration section. Ensure contribution percentages sum to 100% and all members sign.
- The assignment requires at least four pages of main report text, excluding references and the contribution/declaration section.
- Export the final PDF from Overleaf and submit it with the other required project files.

## Important
This is a writing scaffold, not a completed report. It does not claim that retrieval methods, tools, skills, UI features, or experiments have been implemented. Describe only what your group actually builds and measures. The report instructions supplied for the assignment specify the model/API and evaluation deliverables; verify final details against the official course instructions before submission.
