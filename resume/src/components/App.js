import '../index.scss'
import {projects} from '../resumeData'
import {Projects} from './Projects'
import {FrontPage} from './FrontPage'

function App() {
    return (
        <div className="fluid-container">
            <FrontPage></FrontPage>
            <Projects projects={projects}></Projects>
        </div>
    );
}

export default App;
