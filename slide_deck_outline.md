# Slide Deck Outline: Emotional Divergence Between Tech & Public Universities

## Slide 1: Title Slide
**Title:** Multimodal Analysis on Student Mood: Tech vs. Public Universities
**Subtitle:** Contextualizing Emotional Divergence using NLP and Topic Modeling
**Presenter:** [Your Name]

---

## Slide 2: The Core Research Question
**Heading:** Research Question 1
**Content:** 
* "What emotion differences can be caused by the difference between tech university and public university?"
* **The Challenge:** Raw emotion scores tell us *what* students are feeling, but they don't tell us *why*.
* **Our Approach:** Move beyond simple averages by mapping statistically significant emotional differences to unsupervised contextual topic models.

---

## Slide 3: Data & Methodology
**Heading:** A Multi-Layered NLP Pipeline
**Content:**
1. **Data Sources:** Processed Reddit reviews from Georgia Tech (Tech) and UNC (Public).
2. **Emotion Extraction:** 28 granular emotion probabilities extracted per post (GoEmotions framework).
3. **Topic Modeling:** BERTopic applied to uncover latent conversational themes (e.g., "Campus Safety", "Course Registration").
4. **The Synthesis:** Overlaying the emotion probability distributions onto the BERT topics to pinpoint the exact institutional mechanisms driving student mood.

---

## Slide 4: Statistical Rigor – Choosing What to Analyze
**Heading:** Finding Meaningful Divergence
**Content:** 
* **The Method:** We didn't guess which emotions mattered. We ran a **Mann-Whitney U Test** combined with **Cohen’s *d* Effect Size** across all 28 emotions.
* **The Goal:** Find the emotions with the highest magnitude of difference between the two campuses, not just statistical noise.
* **The Results:** 
  * **Curiosity:** Largest effect size favoring the Public University (UNC) (Cohen's d = -0.10, p < 1e-198)
  * **Annoyance:** Largest negative effect size favoring the Tech University (GATECH) (Cohen's d = +0.07, p < 1e-48)

---

## Slide 5: The "What" – A Tale of Two Emotions
**Heading:** Annoyance vs. Curiosity
**Content:**
* **Tech University (GATECH):** The signature differentiating emotion is **negative (Annoyance)**. Students here exhibit significantly higher baseline levels of frustration.
* **Public University (UNC):** The signature differentiating emotion is **exploratory/positive (Curiosity)**. The environment fosters higher engagement and inquisitiveness.
* **The Next Step:** What specific aspects of campus life trigger these exact emotions?

---

## Slide 6: Contextualizing Annoyance (The "Why" for Tech)
**Heading:** Triggers of Annoyance at Tech
**Content:**
* *Mapping the highest average Annoyance scores to specific institutional topics.*
* **Top Annoyance Drivers at GATECH:**
  1. **Campus Politics** (Score: 0.0617)
  2. **Internet & Connectivity** (Score: 0.0603)
  3. **Transportation & Parking** (Score: 0.0572)
  4. **College Football** (Score: 0.0540)
  5. **Dorm Life & Hygiene** (Score: 0.0528)
* **Takeaway:** Annoyance at a tech university is heavily driven by structural, logistical, and infrastructural frictions.

---

## Slide 7: Contextualizing Curiosity (The "Why" for Public)
**Heading:** Triggers of Curiosity at Public
**Content:**
* *Mapping the highest average Curiosity scores to specific institutional topics.*
* **Top Curiosity Drivers at UNC:**
  1. **Academic Resources** (Score: 0.1124)
  2. **Academics & Coursework** (Score: 0.1119)
  3. **Student Organizations** (Score: 0.1080)
  4. **Dining & Food** (Score: 0.1067)
  5. **Campus Housing & Parking** (Score: 0.1043)
* **Takeaway:** Curiosity at a public university is driven by academic and social exploration, navigating the massive breadth of resources and communities available.

---

## Slide 8: Conclusion & Actionable Insights
**Heading:** Answering Research Question 1
**Content:**
* **The Answer:** The tech university environment inherently drives up structural *Annoyance*, while the public university environment fosters exploratory *Curiosity*.
* **The "Why":** Tech universities face distinct friction in digital and physical infrastructure (connectivity, transportation), whereas public universities generate emotional engagement through broad academic and extracurricular ecosystems.
* **Impact:** Institutions can use this multi-modal mapping to target the exact pain points (e.g., upgrading campus Wi-Fi at tech schools) rather than relying on generic "wellness" initiatives.
