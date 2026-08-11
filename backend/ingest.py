import os
import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bharat_data.json"
)

DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json():

    print(f"Reading Bharat data from: {JSON_PATH}")

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"JSON file not found: {JSON_PATH}"
        )

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(data):

    documents = []

    # --------------------------------------------------------
    # PERSONAL
    # --------------------------------------------------------

    personal = data.get("personal", {})

    personal_text = f"""
Bharat Sirmal - Personal Information

Name: {personal.get("name", "")}

Email: {personal.get("email", "")}

Origin: {personal.get("origin", "")}

Current location listed on resume:
{personal.get("current_location_listed_on_resume", "")}

Date of birth:
{personal.get("date_of_birth", {})}

Professional links:
{", ".join(personal.get("professional_links", []))}
"""

    documents.append(
        Document(
            page_content=personal_text.strip(),
            metadata={
                "section": "personal",
                "source": "bharat_data.json"
            }
        )
    )


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    education = data.get("education", {})

    school = education.get(
        "school_education",
        {}
    )

    higher_secondary = education.get(
        "higher_secondary",
        {}
    )

    bachelor = education.get(
        "bachelor",
        {}
    )

    semester_cgpa = bachelor.get(
        "semester_cgpa",
        {}
    )

    education_text = f"""
Bharat Sirmal - Education

School Education:
School: {school.get("school", "")}
Location: {school.get("location", "")}
Grades: {school.get("grades", "")}
Completed Year: {school.get("completed_year", "")}
GPA: {school.get("gpa", "")}/{school.get("gpa_scale", "")}

Higher Secondary Education:
School: {higher_secondary.get("school", "")}
Also listed as: {higher_secondary.get("also_listed_as", "")}
Location: {higher_secondary.get("location", "")}
Grades: {higher_secondary.get("grades", "")}
Completed Year: {higher_secondary.get("completed_year", "")}
GPA: {higher_secondary.get("gpa", "")}/{higher_secondary.get("gpa_scale", "")}
Study Period: {higher_secondary.get("resume_study_period", "")}

Bachelor's Degree:
Degree: {bachelor.get("degree", "")}
Field: {bachelor.get("field_as_provided", "")}
Field in Resume Heading: {bachelor.get("field_in_resume_heading", "")}
College: {bachelor.get("college", "")}
Location: {bachelor.get("location", "")}
Affiliation: {bachelor.get("affiliation", "")}
Study Period: {bachelor.get("study_period", "")}
Resume CGPA: {bachelor.get("resume_cgpa", "")}/{bachelor.get("cgpa_scale", "")}

Semester CGPA:
Semester 1: {semester_cgpa.get("semester_1", "")}
Semester 2: {semester_cgpa.get("semester_2", "")}
Semester 3: {semester_cgpa.get("semester_3", "")}
Semester 4: {semester_cgpa.get("semester_4", "")}
Semester 5: {semester_cgpa.get("semester_5", "")}
Semester 6: {semester_cgpa.get("semester_6", "")}
"""

    documents.append(
        Document(
            page_content=education_text.strip(),
            metadata={
                "section": "education",
                "source": "bharat_data.json"
            }
        )
    )


    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    experience = data.get("experience", {})

    for internship in experience.get(
        "internships",
        []
    ):

        responsibilities = "\n".join(
            f"- {item}"
            for item in internship.get(
                "responsibilities",
                []
            )
        )

        technologies = ", ".join(
            internship.get(
                "technologies",
                []
            )
        )

        experience_text = f"""
Bharat Sirmal - Internship Experience

Position:
{internship.get("position", "")}

Company:
{internship.get("company", "")}

Period:
{internship.get("period", "")}

Responsibilities:
{responsibilities}

Technologies:
{technologies}
"""

        documents.append(
            Document(
                page_content=experience_text.strip(),
                metadata={
                    "section": "experience",
                    "source": "bharat_data.json",
                    "company": internship.get(
                        "company",
                        ""
                    )
                }
            )
        )


    # --------------------------------------------------------
    # PROJECTS
    # --------------------------------------------------------

    projects = data.get(
        "projects",
        []
    )

    for project in projects:

        technologies = ", ".join(
            project.get(
                "technologies",
                []
            )
        )

        work = "\n".join(
            f"- {item}"
            for item in project.get(
                "work",
                []
            )
        )

        project_text = f"""
Bharat Sirmal - Project

Project Name:
{project.get("name", "")}

Technologies:
{technologies}

Work and Contributions:
{work}
"""

        documents.append(
            Document(
                page_content=project_text.strip(),
                metadata={
                    "section": "projects",
                    "source": "bharat_data.json",
                    "project": project.get(
                        "name",
                        ""
                    )
                }
            )
        )


    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    skills = data.get(
        "skills",
        {}
    )

    programming = ", ".join(
        skills.get(
            "programming_languages",
            []
        )
    )

    databases = ", ".join(
        skills.get(
            "data_and_databases",
            []
        )
    )

    productivity = ", ".join(
        skills.get(
            "productivity_and_tools",
            []
        )
    )

    cloud = ", ".join(
        skills.get(
            "cloud_and_platforms",
            []
        )
    )

    additional = ", ".join(
        skills.get(
            "additional_project_and_internship_technologies",
            []
        )
    )

    skills_text = f"""
Bharat Sirmal - Technical Skills

Programming Languages:
{programming}

Data and Databases:
{databases}

Productivity and Tools:
{productivity}

Cloud and Platforms:
{cloud}

Additional Project and Internship Technologies:
{additional}
"""

    documents.append(
        Document(
            page_content=skills_text.strip(),
            metadata={
                "section": "skills",
                "source": "bharat_data.json"
            }
        )
    )


    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    certifications = data.get(
        "certifications_and_achievements",
        []
    )

    for certification in certifications:

        details = "\n".join(
            f"- {item}"
            for item in certification.get(
                "details",
                []
            )
        )

        certification_text = f"""
Bharat Sirmal - Certification

Certification:
{certification.get("name", "")}

Details:
{details}
"""

        documents.append(
            Document(
                page_content=certification_text.strip(),
                metadata={
                    "section": "certifications",
                    "source": "bharat_data.json"
                }
            )
        )


    return documents


# ============================================================
# DELETE OLD DATABASE
# ============================================================

def delete_old_database():

    if os.path.exists(DB_DIR):

        print("Removing old ChromaDB...")

        import shutil

        shutil.rmtree(DB_DIR)

        print("Old ChromaDB removed.")


# ============================================================
# INGEST
# ============================================================

def ingest_data():

    print("=" * 60)
    print("BHARAT AI - DATA INGESTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    data = load_json()

    print("JSON loaded successfully.")

    # --------------------------------------------------------
    # Create logical documents
    # --------------------------------------------------------

    documents = create_documents(data)

    print(
        f"Created {len(documents)} logical documents."
    )

    for i, document in enumerate(documents):

        print(
            f"\n[{i + 1}] "
            f"{document.metadata.get('section')}"
        )

        print(
            document.page_content[:150]
            .replace("\n", " ")
        )

    # --------------------------------------------------------
    # Delete old ChromaDB
    # --------------------------------------------------------

    delete_old_database()

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Create ChromaDB
    # --------------------------------------------------------

    print("Creating ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"Database location: {DB_DIR}"
    )

    print(
        f"Documents stored: {len(documents)}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ingest_data()