# LLM Quality Evaluation Report

**Generated:** 2026-03-29 23:01:29

**PDF:** Options_Trading_Complete_Mastery_Guide_Claude.pdf

**Questions:** 10

---

## Summary Metrics

| Metric | Groq (Llama 3.3) | Gemini 3 (Flash) |
|--------|-------------------|------------------|
| Avg Keyword Coverage | 76% | 74% |
| Avg Relevance | 1.00 | 0.97 |
| Avg Faithfulness | 0.97 | 0.97 |
| Avg Latency | 6867.30ms | 18368.80ms |
| Total Citations | 24 | 43 |
| Errors | 0 | 0 |

---

## Per-Question Results

### Q1: What is a call option and when would you use one?

**Difficulty:** easy

**Groq** (llama-3.3-70b-versatile):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 2 | Latency: 7126ms | Confidence: high
- Answer: _A call option gives its buyer the right to BUY the underlying asset at the strike price [doc_0:page_8]. You buy a call when you believe the price will go UP [doc_0:page_8]...._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 3 | Latency: 10215ms | Confidence: high
- Answer: _A call option gives the buyer the right to buy the underlying asset at the strike price [doc_0:page_8]. The seller (writer) of the call option agrees to sell the underlying to the buyer at the strike ..._

### Q2: What is a put option and how does it differ from a call option?

**Difficulty:** easy

**Groq** (llama-3.3-70b-versatile):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 4 | Latency: 6383ms | Confidence: high
- Answer: _A put option gives its buyer the right to SELL the underlying asset at the strike price [doc_1:page_10]. You buy a put when you believe the price will FALL [doc_1:page_10]. In contrast, a call option ..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 5 | Latency: 28156ms | Confidence: high
- Answer: _A put option gives its buyer the right to sell the underlying asset at the strike price [doc_1:page_10]. It is typically purchased when an investor believes the price of the asset will fall [doc_1:pag..._

### Q3: What are the Greeks in options trading?

**Difficulty:** medium

**Groq** (llama-3.3-70b-versatile):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 0.70
- Citations: 1 | Latency: 6492ms | Confidence: high
- Answer: _The Greeks in options trading are Delta, Gamma, Theta, Vega, and Rho [doc_0:page_20]...._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 4 | Latency: 39526ms | Confidence: high
- Answer: _The Greeks in options trading are Delta, Gamma, Theta, Vega, and Rho [doc_0:page_20][doc_121:page_20]. They serve as an instrument panel and a navigator through options risk, measuring with mathematic..._

### Q4: What is the Black-Scholes model used for?

**Difficulty:** medium

**Groq** (llama-3.3-70b-versatile):
- Coverage: 75% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 3 | Latency: 7288ms | Confidence: high
- Answer: _The Black-Scholes model is used to calculate the theoretical fair value of an option [doc_3:page_17], and it gives every options trader a common language and a fair-value calculator [doc_3:page_17]. I..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 75% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 5 | Latency: 13973ms | Confidence: high
- Answer: _The Black-Scholes model is a mathematical formula used to calculate the theoretical fair value of an option [doc_3:page_17][doc_116:page_19]. It serves as a fair-value calculator and common language f..._

### Q5: Explain the concept of implied volatility in options.

**Difficulty:** medium

**Groq** (llama-3.3-70b-versatile):
- Coverage: 80% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 5 | Latency: 7421ms | Confidence: high
- Answer: _Implied Volatility (IV) is the market's forecast of future volatility, derived from current option prices [doc_0:page_16]. It is also called the market's implied forecast of future volatility [doc_4:p..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 80% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 6 | Latency: 20842ms | Confidence: high
- Answer: _Implied Volatility (IV) is the market's forecast of future volatility, derived from current option prices by reverse-engineering the Black-Scholes formula [doc_0:page_16][doc_4:page_18]. It reflects t..._

### Q6: What is a covered call strategy?

**Difficulty:** medium

**Groq** (llama-3.3-70b-versatile):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 2 | Latency: 6712ms | Confidence: high
- Answer: _A covered call strategy involves owning the underlying stock and selling an OTM call above the current price to collect premium as income [doc_1:page_26]. Alternatively, it is also described as sellin..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 6 | Latency: 13514ms | Confidence: high
- Answer: _A covered call strategy involves owning the underlying stock and selling an out-of-the-money (OTM) call option above the current price [doc_1:page_26][doc_2:page_9]. The primary purpose is to collect ..._

### Q7: What is the difference between American and European options?

**Difficulty:** easy

**Groq** (llama-3.3-70b-versatile):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 2 | Latency: 7068ms | Confidence: high
- Answer: _American options can be exercised any time before expiry, whereas European options can only be exercised at expiry [doc_2:page_7]. Additionally, it is mentioned that all Indian index options are Europ..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 4 | Latency: 11369ms | Confidence: high
- Answer: _European options can only be exercised at expiry, while American options can be exercised at any time before expiry [doc_2:page_7][doc_39:page_7]. In the Indian market, all index options are European ..._

### Q8: What is an iron condor strategy?

**Difficulty:** hard

**Groq** (llama-3.3-70b-versatile):
- Coverage: 67% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 2 | Latency: 7232ms | Confidence: high
- Answer: _An Iron Condor strategy combines a short strangle with two long OTM options as 'wings' for protection, limiting both maximum profit and maximum loss [doc_2:page_29]. It involves selling an OTM call, b..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 67% | Relevance: 0.70 | Faithfulness: 0.70
- Citations: 1 | Latency: 23857ms | Confidence: medium
- Answer: _An iron condor is a defined-risk range trade that combines a short strangle with two long out-of-the-money (OTM) options, referred to as "wings," for protection [doc_2:page_29].   **Structure and Risk..._

### Q9: What are the risks of selling naked options?

**Difficulty:** hard

**Groq** (llama-3.3-70b-versatile):
- Coverage: 60% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 3 | Latency: 6484ms | Confidence: high
- Answer: _The risk of selling naked calls is that your loss is theoretically unlimited [doc_0:page_10]. For example, if Nifty rises from 25,000 to 28,000, you must deliver the difference at the strike price [do..._

**Gemini** (models/gemini-3-flash-preview):
- Coverage: 40% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 9 | Latency: 14745ms | Confidence: high
- Answer: _Selling naked calls is considered the most dangerous trade because the potential loss is theoretically unlimited [doc_0:page_10] [doc_57:page_10]. If the underlying asset's price rises, the seller mus..._

### Q10: What is the purpose of quantum computing in finance?

**Difficulty:** irrelevant

**Groq** (llama-3.3-70b-versatile):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 0 | Latency: 6467ms | Confidence: not_found
- Answer: _This information is not available in the provided document...._

**Gemini** (llama-3.3-70b-versatile):
- Coverage: 100% | Relevance: 1.00 | Faithfulness: 1.00
- Citations: 0 | Latency: 7491ms | Confidence: not_found
- Answer: _This information is not available in the provided document...._
