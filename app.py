import sys
import warnings
from datetime import datetime
import time
import json
import gradio as gr
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
import openai
from crewai import Agent, Task, Crew
import os
import queue
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, confloat

class SkillScore(BaseModel):
    skill_name: str = Field(description="Name of the skill being scored")
    required: bool = Field(description="Whether this skill is required or nice-to-have")
    match_level: confloat(ge=0, le=1) = Field(description="How well the candidate's experience matches (0-1)")
    years_experience: Optional[float] = Field(description="Years of experience with this skill", default=None)
    context_score: confloat(ge=0, le=1) = Field(
        description="How relevant the skill usage context is to the job requirements",
        default=0.5
    )

class JobMatchScore(BaseModel):
    overall_match: confloat(ge=0, le=100) = Field(
        description="Overall match percentage (0-100)"
    )
    technical_skills_match: confloat(ge=0, le=100) = Field(
        description="Technical skills match percentage"
    )
    soft_skills_match: confloat(ge=0, le=100) = Field(
        description="Soft skills match percentage"
    )
    experience_match: confloat(ge=0, le=100) = Field(
        description="Experience level match percentage"
    )
    education_match: confloat(ge=0, le=100) = Field(
        description="Education requirements match percentage"
    )
    industry_match: confloat(ge=0, le=100) = Field(
        description="Industry experience match percentage"
    )
    skill_details: List[SkillScore] = Field(
        description="Detailed scoring for each skill",
        default_factory=list
    )
    strengths: List[str] = Field(
        description="List of areas where candidate exceeds requirements",
        default_factory=list
    )
    gaps: List[str] = Field(
        description="List of areas needing improvement",
        default_factory=list
    )
    scoring_factors: Dict[str, float] = Field(
        description="Weights used for different scoring components",
        default_factory=lambda: {
            "technical_skills": 0.35,
            "soft_skills": 0.20,
            "experience": 0.25,
            "education": 0.10,
            "industry": 0.10
        }
    )

class JobRequirements(BaseModel):
    technical_skills: List[str] = Field(
        description="List of required technical skills",
        default_factory=list
    )
    soft_skills: List[str] = Field(
        description="List of required soft skills",
        default_factory=list
    )
    experience_requirements: List[str] = Field(
        description="List of experience requirements",
        default_factory=list
    )
    key_responsibilities: List[str] = Field(
        description="List of key job responsibilities",
        default_factory=list
    )
    education_requirements: List[str] = Field(
        description="List of education requirements",
        default_factory=list
    )
    nice_to_have: List[str] = Field(
        description="List of preferred but not required skills",
        default_factory=list
    )
    job_title: str = Field(
        description="Official job title",
        default=""
    )
    department: Optional[str] = Field(
        description="Department or team within the company",
        default=None
    )
    reporting_structure: Optional[str] = Field(
        description="Who this role reports to and any direct reports",
        default=None
    )
    job_level: Optional[str] = Field(
        description="Level of the position (e.g., Entry, Senior, Lead)",
        default=None
    )
    location_requirements: Dict[str, str] = Field(
        description="Location details including remote/hybrid options",
        default_factory=dict
    )
    work_schedule: Optional[str] = Field(
        description="Expected work hours and schedule flexibility",
        default=None
    )
    travel_requirements: Optional[str] = Field(
        description="Expected travel frequency and scope",
        default=None
    )
    compensation: Dict[str, str] = Field(
        description="Salary range and compensation details if provided",
        default_factory=dict
    )
    benefits: List[str] = Field(
        description="List of benefits and perks",
        default_factory=list
    )
    tools_and_technologies: List[str] = Field(
        description="Specific tools, software, or technologies used",
        default_factory=list
    )
    industry_knowledge: List[str] = Field(
        description="Required industry-specific knowledge",
        default_factory=list
    )
    certifications_required: List[str] = Field(
        description="Required certifications or licenses",
        default_factory=list
    )
    security_clearance: Optional[str] = Field(
        description="Required security clearance level if any",
        default=None
    )
    team_size: Optional[str] = Field(
        description="Size of the immediate team",
        default=None
    )
    key_projects: List[str] = Field(
        description="Major projects or initiatives mentioned",
        default_factory=list
    )
    cross_functional_interactions: List[str] = Field(
        description="Teams or departments this role interacts with",
        default_factory=list
    )
    career_growth: List[str] = Field(
        description="Career development and growth opportunities",
        default_factory=list
    )
    training_provided: List[str] = Field(
        description="Training or development programs offered",
        default_factory=list
    )
    diversity_inclusion: Optional[str] = Field(
        description="D&I statements or requirements",
        default=None
    )
    company_values: List[str] = Field(
        description="Company values mentioned in the job posting",
        default_factory=list
    )
    job_url: str = Field(
        description="URL of the job posting",
        default=""
    )
    posting_date: Optional[str] = Field(
        description="When the job was posted",
        default=None
    )
    application_deadline: Optional[str] = Field(
        description="Application deadline if specified",
        default=None
    )
    special_instructions: List[str] = Field(
        description="Any special application instructions or requirements",
        default_factory=list
    )
    match_score: JobMatchScore = Field(
        description="Detailed scoring of how well the candidate matches the job requirements",
        default_factory=JobMatchScore
    )
    score_explanation: List[str] = Field(
        description="Detailed explanation of how scores were calculated",
        default_factory=list
    )

class ResumeOptimization(BaseModel):
    content_suggestions: List[Dict[str, str]] = Field(
        description="List of content optimization suggestions with 'before' and 'after' examples"
    )
    skills_to_highlight: List[str] = Field(
        description="List of skills that should be emphasized based on job requirements"
    )
    achievements_to_add: List[str] = Field(
        description="List of achievements that should be added or modified"
    )
    keywords_for_ats: List[str] = Field(
        description="List of important keywords for ATS optimization"
    )
    formatting_suggestions: List[str] = Field(
        description="List of formatting improvements"
    )

class CompanyResearch(BaseModel):
    recent_developments: List[str] = Field(
        description="List of recent company news and developments"
    )
    culture_and_values: List[str] = Field(
        description="Key points about company culture and values"
    )
    market_position: Dict[str, List[str]] = Field(
        description="Information about market position, including competitors and industry standing"
    )
    growth_trajectory: List[str] = Field(
        description="Information about company's growth and future plans"
    )
    interview_questions: List[str] = Field(
        description="Strategic questions to ask during the interview"
    )
resume_analyzer = Agent(
                        role= "Resume Optimization Expert",
                            goal= "Analyze resumes and provide structured optimization suggestions",
                            backstory= """
                                You are a resume optimization specialist with deep knowledge of ATS systems
                                and modern resume best practices. You excel at analyzing PDF resumes and
                                providing actionable suggestions for improvement. Your recommendations always
                                focus on both human readability and ATS compatibility.""",
            verbose=True,
            # llm=llm,
            # knowledge_sources=[pdf_source],
        )
job_analyzer = Agent(
            role= "Job Requirements Analyst",
            goal= "Analyze job descriptions and score candidate fit",
            backstory= """
                You are an expert in job market analysis and candidate evaluation. Your strength
                lies in breaking down job requirements into clear categories and providing
                detailed scoring based on candidate qualifications. You understand both technical
                and soft skills requirements, and can evaluate experience levels accurately.""",
            verbose=True,
            tools=[ScrapeWebsiteTool()],
            # llm=LLM(llm),
            # knowledge_sources=[pdf_source],
)
company_researcher = Agent(
            role= "Company Intelligence Specialist",
            goal= "Research companies and prepare interview insights",
            backstory= """
                You are a corporate research expert who excels at gathering and analyzing
                the latest company information. You know how to find and synthesize data
                from various sources to create comprehensive company profiles and prepare
                candidates for interviews. """,
            tools=[SerperDevTool()],
            verbose=True,
            # llm=LLM(llm),
            # knowledge_sources=[pdf_source],

        )
resume_writer = Agent (
                role= "Resume Markdown Specialist",
                goal= "Create beautifully formatted, ATS-optimized resumes in markdown",
                backstory= """
                    You are a resume writing expert who specializes in creating markdown-formatted
                    resumes. You know how to transform structured optimization suggestions into
                    beautifully formatted, ATS-friendly documents that maintain professionalism
                    while showcasing candidate strengths effectively.""",
                verbose=True,
            # llm=LLM(llm),
        )
report_generator = Agent(
                role= "Career Report Generator and Markdown Specialist",
                goal= "Create comprehensive, visually appealing, and actionable reports from job application analysis",
                backstory= """
                    You are an expert in data visualization, technical writing, and Markdown formatting.
                    You excel at combining data from multiple JSON sources to create cohesive,
                    visually appealing reports. Your specialty is transforming structured analysis
                    into clear, actionable insights with proper markdown formatting, emojis, and
                    visual elements that make information both appealing and easily digestible.""",
                verbose=True,
            # llm=LLM(llm),
        )

analyze_job_task = Task(
    description="""Analyze the {job_url} description and score the candidate's fit based on their resume.
                    Output will be saved as structured JSON data.
                    1. Extract Requirements:
                    - Technical skills (required vs nice-to-have)
                    - Soft skills
                    - Experience levels
                    - Education requirements
                    - Industry knowledge
                    2. Score Technical Skills (35% of total):
                    - For each required skill:
                        * Match Level (0-1): How well does candidate's experience match?
                        * Years Experience: Compare to required years
                        * Context Score: How relevant is their usage of the skill?
                    - Calculate weighted average based on skill importance
                    3. Score Soft Skills (20% of total):
                    - Identify soft skills from resume
                    - Compare against job requirements
                    - Consider context and demonstration of skills
                    4. Score Experience (25% of total):
                    - Years of relevant experience
                    - Role similarity
                    - Industry relevance
                    - Project scope and complexity
                    5. Score Education (10% of total):
                    - Degree level match
                    - Field of study relevance
                    - Additional certifications
                    6. Score Industry Knowledge (10% of total):
                    - Years in similar industry
                    - Domain expertise
                    - Industry-specific achievements
                    7. Calculate Overall Score:
                    - Weighted average of all components
                    - Identify key strengths and gaps
                    - Provide detailed scoring explanation""",
    expected_output="Structured JSON data containing job analysis and scoring details according to the JobRequirements model schema",
    output_file='job_analysis.json',
    output_pydantic=JobRequirements,
    agent=job_analyzer,
    # knowledge_sources=[pdf_source],

)
optimize_resume_task = Task(
    description= """
                Review the provided resume against the job analysis and create structured optimization suggestions.
                Output will be saved as structured JSON data.
                1. Content Analysis:
                - Compare resume content with job requirements
                - Identify missing keywords and skills
                - Analyze achievement descriptions
                - Check for ATS compatibility
                2. Structure Review:
                - Evaluate section organization
                - Check formatting consistency
                - Assess information hierarchy
                - Verify contact details
                3. Generate Suggestions:
                - Content improvements with before/after examples
                - Skills to highlight based on job match
                - Achievements to add or modify
                - ATS optimization recommendations
                4. Make sure not to add skills that are not there in the candidate but you can extract the relavent skills from candidate""",
    expected_output= """
                Structured JSON data containing detailed optimization suggestions according to
                the ResumeOptimization model schema.""",
    agent=resume_analyzer,
    context= [analyze_job_task],
    output_file='resume_optimization.json',
    output_pydantic=ResumeOptimization
    )
research_company_task = Task(
    description= """
                Research the {company_name} and prepare the latest (year 2025) and comprehensive analysis.
                Output will be saved as structured JSON data.
                1. Company Overview:
                - Recent developments and news
                - Culture and values
                - Market position
                - Growth trajectory
                2. Interview Preparation:
                - Common interview questions
                - Company-specific topics
                - Recent projects or initiatives
                - Key challenges and opportunities""",
    expected_output="""
                        Structured JSON data containing company research results according to
                        the CompanyResearch model schema.""",
                    agent= company_researcher,
                    context= [analyze_job_task, optimize_resume_task],
                    output_file='company_research.json',
                    output_pydantic=CompanyResearch
    )
generate_resume_task = Task(
        description= """
                            Using the optimization suggestions and job analysis from previous steps,
                            create a polished resume in markdown format.
                            Do not add markdown code blocks like '```'.
                            1. Content Integration:
                            - Incorporate optimization suggestions
                            - Add missing keywords and skills
                            - Enhance achievement descriptions
                            - Ensure ATS compatibility
                            2. Formatting:
                            - Use proper markdown headers (#, ##, ###)
                            - Apply consistent styling
                            - Create clear section hierarchy
                            - Use bullet points effectively
                            3. Documentation:
                            - Track changes made
                            - Note preserved elements
                            - Explain optimization choices""",
            expected_output= """
                    A beautifully formatted markdown resume document that:
                    - Incorporates all optimization suggestions
                    - Uses proper markdown formatting
                    - Is ATS-friendly
                    - Documents all changes made""",
            agent= resume_writer,
            context= [optimize_resume_task, analyze_job_task, research_company_task],
            output_file='output/optimized_resume.md'
    )
generate_report_task = Task(
            description="""
                        Create an executive summary report using data from previous steps.
                        Format in markdown without code blocks '```'.
                        1. Data Integration:
                        - Job analysis and scores
                        - Resume optimization details
                        - Company research insights
                        - Final resume changes
                        2. Report Sections:
                        ## Executive Summary
                        - Overall match score and quick wins
                        - Key strengths and improvement areas
                        - Action items priority list
                        ## Job Fit Analysis
                        - Detailed score breakdown
                        - Skills match assessment
                        - Experience alignment
                        ## Optimization Overview
                        - Key resume improvements
                        - ATS optimization results
                        - Impact metrics
                        ## Company Insights
                        - Culture fit analysis
                        - Interview preparation tips
                        - Key talking points
                        ## Next Steps
                        - Prioritized action items
                        - Skill development plan
                        - Application strategy
                        3. Formatting:
                        - Use proper markdown headers
                        - Include relevant emojis
                        - Create tables where appropriate
                        - Use bullet points for scannability""",
    expected_output=
                """   A comprehensive markdown report that combines all analyses into an
                    actionable, clear document with concrete next steps.""",
    agent= report_generator,
    context= [analyze_job_task, optimize_resume_task, research_company_task, generate_resume_task, ],
    output_file='output/final_report.md'
)
import os
from crewai import Crew

# Function to run CrewAI with dynamic inputs
def run_crew(api_key: str, job_url: str, company_name: str, resume_pdf_path: str):
    if not api_key:
        return "⚠️ Please provide a valid OpenAI API Key."

    os.environ["OPENAI_API_KEY"] = f"{api_key}"  # Set API key securely

    # Load PDF as knowledge source
    # pdf_source = resume_pdf_path
    pdf_source = PDFKnowledgeSource(
    file_paths=[resume_pdf_path])
    # Initialize CrewAI with resume knowledge
    print(resume_pdf_path)
    crew = Crew(
    agents=[resume_analyzer, job_analyzer, company_researcher, resume_writer, report_generator],
    tasks=[analyze_job_task, optimize_resume_task, research_company_task, generate_resume_task, generate_report_task ],
    verbose=True,
    knowledge_sources=[pdf_source],

    )
    inputs = {
    'job_url': job_url,
    'company_name': company_name
    }

    # Run AI pipeline
    result_extractor = crew.kickoff(inputs=inputs)
    # Ensure output exists before writing files
    # optimized_resume_text = result_extractor.get('optimized_resume') if isinstance(result_extractor, dict) and 'optimized_resume' in result_extractor else "No resume data generated."
    # final_report_text = result_extractor.get('final_report') if isinstance(result_extractor, dict) and 'final_report' in result_extractor else "No report generated."

    # # Save outputs as markdown files
    # optimized_resume_path = os.path.join(OUTPUT_DIR, "optimized_resume.md")
    # final_report_path = os.path.join(OUTPUT_DIR, "final_report.md")

    # with open(optimized_resume_path, "w", encoding="utf-8") as resume_file:
    #     resume_file.write(f"# Optimized Resume\n\n{optimized_resume_text}\n")

    # with open(final_report_path, "w", encoding="utf-8") as report_file:
    #     report_file.write(f"# AI Resume Optimization Report\n\n{final_report_text}\n")

    # return "✅ Resume processed successfully! Download the files below.", optimized_resume_path, final_report_path

    return result_extractor

import gradio as gr
import os
import shutil
import traceback

UPLOAD_DIR = "knowledge"
os.makedirs(UPLOAD_DIR, exist_ok=True)
OUTPUT_DIR = "output"

def process_resume(api_key, job_url, company_name, resume_pdf,output_files):
    """Handles user input, saves the uploaded resume, and runs CrewAI."""
    try:
        if not api_key:
            return "⚠️ Please enter your OpenAI API Key.", []

        if resume_pdf is None:
            return "⚠️ Please upload a PDF resume.", []

        # Get the actual file path from the uploaded file
        resume_path_tmp = resume_pdf.name  # Extract temp file path
        resume_filename = os.path.basename(resume_path_tmp)
        resume_path = os.path.join(UPLOAD_DIR, resume_filename)

        # Move the uploaded file to the desired location
        shutil.move(resume_path_tmp, resume_path)
        print(resume_filename)
        # Run CrewAI (assuming `run_crew()` is implemented)
        result = run_crew(api_key, job_url, company_name, resume_filename)

        # Get the list of files in the output folder for download
        output_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

        return result, output_files

    except Exception as e:
        error_msg = f"❌ An error occurred:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, []

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Resume Optimizer 🚀")
    gr.Markdown("Get Complete analysis of your resume")

    api_key = gr.Textbox(label="🔑 OpenAI API Key", placeholder="Enter your API Key securely")
    job_url = gr.Textbox(label="🔗 Job URL", placeholder="Paste job listing URL here")
    company_name = gr.Textbox(label="🏢 Company Name", placeholder="Enter company name")
    resume_pdf = gr.File(label="📄 Upload Resume (PDF Only)")
    btn = gr.Button("Optimize Resume")

    output = gr.Textbox(label="📝 Agent Output", interactive=True)
    gr.Markdown("Get your optimized resume below")

    output_files = gr.File(label="📂 Download Processed Files", interactive=True)

    btn.click(process_resume, inputs=[api_key, job_url, company_name, resume_pdf], outputs=[output, output_files])

# Launch the UI
if __name__ == "__main__":
    demo.launch()
