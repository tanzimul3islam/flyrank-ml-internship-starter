## Voice card

**Direct, curious, practical, honest, plain, specific.**

Standing instruction for this document and this project: write short, natural sentences. Use concrete details from the actual work. Cut buzzwords and any claim the evidence doesn't support. Distinguish Pinn's contributions from AI assistance, and finished work from future plans. Don't invent motivations, decisions, experience, or results. Keep the writing aimed at one reader — a junior data/ML hiring manager — with one action: email Pinn about a role.

## Bio

I'm building my data and ML portfolio through the FlyRank internship. My first project uses search-performance data to frame a practical question: which pages should an editor review first?

## FlyRank — Which pages deserve a closer look?

*An early-stage analysis of 30,000 pages, with AI assistance.*

### The problem

An editor can't give every page the same attention. But a drop in search impressions doesn't automatically mean a page needs rewriting — it might reflect seasonal demand, traffic moving to another page, or noise.

This project asks which pages deserve review first. A poor recommendation wastes an editor's time and could lead to an unnecessary change. Missing a useful candidate could delay attention to a real problem.

### What I did and decided

I used AI assistance to turn the internship brief into a research question and an executed notebook. The assistant drafted the framing and ran the calculations; the notebook documents the assumptions, code, and outputs so the work can be checked.

The analysis takes one page as its unit. It starts with pages that received at least 500 search impressions over 90 days, then looks at recorded impression decline and time since the last update. The 500-impression cutoff and a 20-page review budget are provisional assumptions, not requirements from a real editorial team.

The starting rule is simple: among eligible pages, review the longest-unupdated first. The proposed output is a queue with supporting measurements and reasons for review. A person would still decide whether to edit, investigate, or monitor each page.

### What came of it

The notebook found:

| Filter | Pages |
| --- | --- |
| Met the 500-impression floor | 16,726 of 30,000 |
| ...and had a recorded downward impression trend | 9,961 of 16,726 |
| ...and hadn't been updated in at least 90 days | 4,053 of 9,961 |

That's enough potential review work to make prioritization worth investigating. It doesn't establish that those pages need a refresh.

The immediate result is a completed, executed notebook committed to the project repository. No model was trained for this assignment, no editor has validated the proposed queue, and no traffic improvement has been measured.

**Next time:** I'd seek independent editorial judgments and test the ranking on held-out evidence before calling it useful in practice.

[View the executed notebook and its calculations](https://github.com/tanzimul3islam/flyrank-ml-internship-starter/blob/779a697/work/notebooks/w01_research_question.ipynb)

## Contact

Hiring for a junior data or ML role? Email me at [tanzimul3islam@gmail.com](mailto:tanzimul3islam@gmail.com) to talk about this work and where I could contribute.

## Before / after

**Generic AI line:** "I leveraged cutting-edge machine learning to unlock actionable insights and drive measurable SEO growth."

**Edited version:** "With AI assistance, I explored which pages an editor might review first. The notebook identified 4,053 pages with at least 500 impressions, a recorded downward trend, and no update in at least 90 days; whether editing them would help is still untested."

The edit replaces an unsupported growth claim with the work actually completed, the selection criteria, and the unanswered question.
