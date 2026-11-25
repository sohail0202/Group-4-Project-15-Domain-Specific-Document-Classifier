# Project 15 plan

Group 4 will classify English consumer finance complaints by product.
The deliverables cover Project 15 and stay in this workspace.
The capstone PDF defines the technical and presentation requirements, and the two assignment photos identify this team's members.

## Data and evaluation

- Collect public complaint narratives from the CFPB and keep the product label, complaint ID and date.
- Normalize text and remove empty, short, duplicate and conflicting narratives before splitting.
- Use disjoint training, validation and test sets, and show class balance and document length in the EDA.
- Fit a TF-IDF baseline on the training set.
- Fine-tune a compact pretrained BERT model on the same training set and select the checkpoint using validation macro F1.
- Test a validation-tuned blend of transformer and baseline scores for the app.
- Report test accuracy, macro precision, macro recall, macro F1, per-class scores and a confusion matrix.

## Application

- Serve an upload and paste interface on localhost.
- Support TXT, PDF with a text layer, and DOCX.
- Display the predicted product and class scores, with a review cue for uncertain predictions.
- Offer a short extractive summary drawn from the complaint.
- Use a forest-green sidebar for samples and model notes, and an off-white workspace with a confidence dial.
- Check all five categories, keyboard controls, file selection, drag and drop, error recovery and mobile sizing in a browser.

## Submission

- Include the source, trained checkpoint, dataset, training notebook, analysis, setup guide, and project report.
- Create a PowerPoint with the team roster, method, measured results, application screenshot, demo steps and limitations.
- Package the project for local handoff.

| Roll number | Name |
| --- | --- |
| 2401201036 | Drishti Jain |
| 2401201037 | Neha |
| 2401201038 | Yash Lamba |
| 2401201039 | Teesha |
| 2401201040 | Sohail Khan |
| 2401201041 | Parth Behera |
