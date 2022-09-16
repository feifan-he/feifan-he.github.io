import {marked} from 'marked';
import parse from 'html-react-parser'
import {educationMarkdown, certificationsMarkdown, languagesMarkdown} from '../resumeData'


export function EducationAndSkills(props) {
    return (<div className='container-fluid education-and-skills'>
        <div className="row">
            {
                [
                    ['education', './imgs/general/education.png', 'Education', educationMarkdown],
                    ['certifications', './imgs/general/certificate.png', 'Courses and Certifications', certificationsMarkdown],
                    ['skills', './imgs/general/programing.png', 'Skills', languagesMarkdown],
                ].map((category) => {
                    const [className, icon, title, markdownContent] = category;
                    return (<div className='col col-xxl-4 col-12 col-lg-6 category-container'>
                        <hr className='h-div'/>
                        <h1 id={className}>
                            <img className="title-icon text-center" src={icon} alt=""></img>
                            {title}
                        </h1>
                        <div className={className + ' category'}>
                        {parse(marked.parse(markdownContent))}
                        </div>
                    </div>)
                })
            }
            <div className='col-12 text-center ending'>
                I'm actively searching for 2023 summer SDE internship opportunities, <br/>
                please feel free to <a href='mailto:feifanhe@brandeis.edu'>contact me</a> if there's any opening positions.
                <br/>
                <img src='imgs/general/emoji.png'/>
            </div>
        </div>

    </div>)
}