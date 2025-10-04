import pdfplumber
import re
import spacy
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Latent Semantic Analysis Summarizer

# Load SpaCy transformer model for better NER
nlp = spacy.load("en_core_web_trf")

# ---------- Step 1: Read PDF ----------
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# ---------- Step 2: Extract Parties ----------
def extract_parties(text):
    parties = {"Plaintiff": None, "Defendant": None, "Judge": None, "Court": None}
    plaintiff_match = re.search(r'Plaintiff\s*:\s*(.*)', text, re.IGNORECASE)
    defendant_match = re.search(r'Defendant\s*:\s*(.*)', text, re.IGNORECASE)
    judge_match = re.search(r'Before\s*:\s*(.*)', text, re.IGNORECASE)
    court_match = re.search(r'In the\s*(.*Court.*)', text, re.IGNORECASE)

    if plaintiff_match:
        parties["Plaintiff"] = plaintiff_match.group(1).strip()
    if defendant_match:
        parties["Defendant"] = defendant_match.group(1).strip()
    if judge_match:
        parties["Judge"] = judge_match.group(1).strip()
    if court_match:
        parties["Court"] = court_match.group(1).strip()

    # Fallback using NER
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
    if orgs:
        if not parties["Plaintiff"]:
            parties["Plaintiff"] = orgs[0]
        if not parties["Defendant"] and len(orgs) > 1:
            parties["Defendant"] = orgs[1]

    return parties

# ---------- Step 3: Extract Dates ----------
def extract_dates(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ == "DATE"]))

# ---------- Step 4: Extract Clauses ----------
def extract_clauses(text):
    clauses = {}
    clause_patterns = {
        "Jurisdiction": r'(jurisdiction|venue|applicable law|governing law)',
        "Confidentiality": r'(confidentiality|non-disclosure|secrecy)',
        "Intellectual Property": r'(intellectual property|IP|copyright|trademark|patent)',
        "Indemnity": r'(indemnity|indemnification|hold harmless)',
        "Force Majeure": r'(force majeure|act of god|unforeseen events)',
        "Dispute Resolution": r'(arbitration|mediation|dispute resolution)',
        "Termination / Liability": r'(termination|liability|breach|penalty)',
        "Payment / Obligations": r'(payment|fee|compensation|obligation|deliver|provide)',
        "Warranties": r'(warranty|guarantee|representation)',
        "Compliance": r'(compliance|lawful|regulation|statutory)',
        "Miscellaneous": r'(entire agreement|severability|assignment|notices)'
    }
    for clause_name, pattern in clause_patterns.items():
        matches = re.findall(pattern + r'.*?(?:\.|\n)', text, re.IGNORECASE)
        if matches:
            clauses[clause_name] = matches
    return clauses

# ---------- Step 5: Extract Legal References ----------
def extract_legal_references(text):
    references = re.findall(r'(Section\s\d+[A-Za-z]*|Article\s\d+[A-Za-z]*|Clause\s\d+[A-Za-z]*|Act\s[\w\s]+|Code\s[\w\s]+)', text, re.IGNORECASE)
    return list(set(references))

# ---------- Step 6: Summarization ----------
def generate_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    summary = " ".join([str(s) for s in summary_sentences])
    return summary if summary else text[:500] + "..."

# ---------- Step 7: Identify Core of Document ----------
def identify_core(text):
    # Look for high-level keywords to infer the main theme
    themes = {
        "Contract": r'(agreement|contract|obligations|parties)',
        "Judgment / Court Case": r'(judgment|appeal|verdict|court|bench|trial)',
        "Criminal Law": r'(crime|penal|punishment|imprisonment|charges)',
        "Civil Law": r'(civil|compensation|damages|injunction)',
        "Corporate / Business Law": r'(shareholder|merger|acquisition|company law)',
        "Property / Real Estate": r'(lease|property|ownership|mortgage|land)',
        "Employment / Labor": r'(employment|employee|employer|labor|workplace)',
        "Family Law": r'(marriage|divorce|custody|maintenance)',
        "Taxation": r'(tax|income tax|GST|duty|levy)'
    }
    for theme, pattern in themes.items():
        if re.search(pattern, text, re.IGNORECASE):
            return theme
    return "General Legal Document"

# ---------- Step 8: Main Pipeline ----------
def legal_document_understander(pdf_path):
    text = read_pdf(pdf_path)
    parties = extract_parties(text)
    dates = extract_dates(text)
    clauses = extract_clauses(text)
    legal_refs = extract_legal_references(text)
    summary = generate_summary(text)
    core = identify_core(text)

    output = f"""
CORE OF DOCUMENT: {core}

1. Parties:
   - Plaintiff: {parties.get('Plaintiff', 'N/A')}
   - Defendant: {parties.get('Defendant', 'N/A')}
   - Judge: {parties.get('Judge', 'N/A')}
   - Court: {parties.get('Court', 'N/A')}

2. Key Clauses & Obligations:
"""
    for clause_type, clause_list in clauses.items():
        output += f"\n   {clause_type}:\n"
        for clause in clause_list:
            output += f"      - {clause.strip()}\n"

    output += "\n3. Important Dates:\n"
    for date in dates:
        output += f"   - {date}\n"

    output += "\n4. Legal References:\n"
    for ref in legal_refs:
        output += f"   - {ref}\n"

    output += f"\n5. Summary:\n   - {summary}\n"

    return output

# ---------- Step 9: Run ----------
if __name__ == "__main__":
    pdf_file = r"C:\Users\praji\Downloads\Shri_Krishan_vs_The_Kurukshetra_University_on_17_November_1975.PDF"
    result = legal_document_understander(pdf_file)
    print(result)
