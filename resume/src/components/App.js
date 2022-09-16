import '../index.scss'
import {projects} from '../resumeData'
import {Projects} from './Projects'
import {FrontPage} from './FrontPage'
import {EducationAndSkills} from './EducationAndSkills'

function App() {
    return (
        <div className="container-fluid">
            <FrontPage></FrontPage>
            <Projects projects={projects}></Projects>
            <EducationAndSkills></EducationAndSkills>
            <br/><br/>
        </div>
    );
}

export default App;
