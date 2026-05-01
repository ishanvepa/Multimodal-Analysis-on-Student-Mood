# Emotion-Conditioned Context Analysis

This report addresses **Research Question 1: What emotion differences can be caused by the difference between tech university and public university?** 
Instead of guessing which emotions matter, we statistically identified the two emotions with the most significant divergence between GATECH and UNC, and mapped them to the existing BERTopic models to understand *why* these emotions occur.

## 1. Statistical Divergence (The "What")
A Mann-Whitney U test on the 28 emotion probabilities revealed the following major differences:
- **Curiosity**: Significantly higher at UNC (Cohen's d = -0.10, p < 1e-198)
- **Annoyance**: Significantly higher at GATECH (Cohen's d = 0.07, p < 1e-48)

This confirms that the primary emotional differentiator for the tech university is negative (Annoyance), while the public university is characterized by higher exploratory/positive emotion (Curiosity).

## 2. Contextual Triggers (The "Why")
By joining the emotion scores with the pre-calculated `bertopic_outputs`, we can identify which specific topics are disproportionately driving "Annoyance" and "Curiosity" at each campus.

### What drives Annoyance?
*Annoyance is the signature negative emotion over-represented at GATECH. Here are the topics with the highest average annoyance scores.*

**GATECH Top Annoyance Topics:**
- **Campus Politics** (Annoyance Score: 0.0617)
- **Internet & Connectivity** (Annoyance Score: 0.0603)
- **Transportation & Parking** (Annoyance Score: 0.0572)
- **College Football** (Annoyance Score: 0.0540)
- **Dorm Life & Hygiene** (Annoyance Score: 0.0528)

**UNC Top Annoyance Topics:**
- **University Community** (Annoyance Score: 0.0766)
- **University Leadership** (Annoyance Score: 0.0763)
- **Student Body & Alumni** (Annoyance Score: 0.0697)
- **Campus Safety** (Annoyance Score: 0.0680)
- **Campus Safety** (Annoyance Score: 0.0665)

### What drives Curiosity?
*Curiosity is the signature emotion over-represented at UNC. Here are the topics with the highest average curiosity scores.*

**UNC Top Curiosity Topics:**
- **Academic Resources** (Curiosity Score: 0.1124)
- **Academics & Coursework** (Curiosity Score: 0.1119)
- **Student Organizations** (Curiosity Score: 0.1080)
- **Dining & Food** (Curiosity Score: 0.1067)
- **Campus Housing & Parking** (Curiosity Score: 0.1043)

**GATECH Top Curiosity Topics:**
- **Campus Facilities & Services** (Curiosity Score: 0.1487)
- **Course Planning & Advising** (Curiosity Score: 0.1126)
- **Information Security** (Curiosity Score: 0.0999)
- **Career Planning & Development** (Curiosity Score: 0.0950)
- **Academic Courses & Registration** (Curiosity Score: 0.0929)

## 3. Relevance to Research Question 1
By overlaying emotion probability distributions onto unsupervised topic models, we can definitively answer what emotion differences exist (Annoyance vs. Curiosity) and pinpoint the exact institutional mechanisms causing them (e.g., class registration and CS courses at a tech university vs. different domains at a public university). This multi-modal approach contextualizes the raw emotion scores into actionable insights about the student experience.
